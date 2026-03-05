# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS


@DATASETS.register_module()
class CDQAGDataset(BaseDataset):
    """Dataset for the QAG-360K change-detection question-answering benchmark
    (VisTA: Show Me What and Where has Changed).

    Each sample contains:
    - A pair of temporal remote sensing images (``img_path``, ``img_path2``).
    - A binary change mask (``seg_map_path``).
    - A natural-language question about the change (``question``).
    - A 23-class answer label (``answer``).

    **Annotation file format**

    The annotation file is a JSON Lines file (``.jsonl``) or a plain JSON
    file containing a list of records.  Each record must have the following
    keys:

    .. code-block:: json

        {
            "img_path":  "A/00001.png",
            "img_path2": "B/00001.png",
            "seg_map_path": "label/00001.png",
            "question": "What type of land cover change occurred?",
            "answer": 3
        }

    Paths may be relative to ``data_root`` or absolute.

    The ``question`` field is stored as a raw text string in metainfo and
    tokenised by the model at training/inference time using the model's
    ``context_length``, following the same pattern as ``jsonA``/``jsonB``
    in the base ``ChangeCLIP`` pipeline.

    Args:
        ann_file (str): Path to the annotation file (JSON or JSON-Lines).
        metainfo (dict, optional): Meta information, e.g. ``classes`` and
            ``palette``. Defaults to None.
        data_root (str, optional): Root directory for relative paths.
            Defaults to None.
        data_prefix (dict, optional): Prefix paths for image/label files.
            Defaults to ``dict(img_path='', img_path2='',
            seg_map_path='')``.
        img_suffix (str): Image file suffix. Defaults to ``'.png'``.
        img_suffix2 (str): Second image file suffix. Defaults to ``'.png'``.
        seg_map_suffix (str): Segmentation map suffix. Defaults to
            ``'.png'``.
        filter_cfg (dict, optional): Config for filtering data.
            Defaults to None.
        indices (int or Sequence[int], optional): Subset of data to use.
            Defaults to None.
        serialize_data (bool): Whether to serialise data for shared memory.
            Defaults to True.
        pipeline (list): Processing pipeline. Defaults to ``[]``.
        test_mode (bool): Whether in test mode. Defaults to False.
        lazy_init (bool): Skip loading annotations on construction.
            Defaults to False.
        max_refetch (int): Maximum retries for invalid samples.
            Defaults to 1000.
        ignore_index (int): Label index to ignore. Defaults to 255.
        reduce_zero_label (bool): Treat label 0 as ignored.
            Defaults to False.
        backend_args (dict, optional): Backend file I/O arguments.
            Defaults to None.
    """

    METAINFO: dict = dict(
        classes=('background', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(
                     img_path='', img_path2='', seg_map_path=''),
                 img_suffix: str = '.png',
                 img_suffix2: str = '.png',
                 seg_map_suffix: str = '.png',
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = False,
                 backend_args: Optional[dict] = None) -> None:

        self.img_suffix = img_suffix
        self.img_suffix2 = img_suffix2
        self.seg_map_suffix = seg_map_suffix
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))

        # Get label map for custom classes
        new_classes = self._metainfo.get('classes', None)
        self.label_map = self.get_label_map(new_classes)
        self._metainfo.update(
            dict(
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label))

        # Update palette based on label map or generate palette
        updated_palette = self._update_palette()
        self._metainfo.update(dict(palette=updated_palette))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

        if test_mode:
            assert self._metainfo.get('classes') is not None, \
                'dataset metainfo `classes` should be specified when testing'

    @classmethod
    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        """Require label mapping.

        Args:
            new_classes (list, tuple, optional): New class names.
                Defaults to None.

        Returns:
            dict or None: Mapping from old to new class indices, or None.
        """
        old_classes = cls.METAINFO.get('classes', None)
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):
            label_map = {}
            if not set(new_classes).issubset(cls.METAINFO['classes']):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in METAINFO.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        return None

    def _update_palette(self) -> list:
        """Update palette after loading metainfo.

        Returns:
            list: Updated colour palette.
        """
        palette = self._metainfo.get('palette', [])
        classes = self._metainfo.get('classes', [])
        if len(palette) == len(classes):
            return palette
        if len(palette) == 0:
            state = np.random.get_state()
            np.random.seed(42)
            new_palette = np.random.randint(
                0, 255, size=(len(classes), 3)).tolist()
            np.random.set_state(state)
        elif len(palette) >= len(classes) and self.label_map is not None:
            new_palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    new_palette.append(palette[old_id])
            new_palette = type(palette)(new_palette)
        else:
            raise ValueError('palette does not match classes '
                             f'as metainfo is {self._metainfo}.')
        return new_palette

    def _resolve_path(self, prefix: str, rel_path: str) -> str:
        """Return an absolute path by joining prefix and rel_path.

        If rel_path is already absolute the prefix is ignored.

        Args:
            prefix (str): Directory prefix.
            rel_path (str): Relative (or absolute) path.

        Returns:
            str: Resolved path.
        """
        if osp.isabs(rel_path):
            return rel_path
        return osp.join(prefix, rel_path)

    def load_data_list(self) -> List[dict]:
        """Load annotations from the JSON/JSON-Lines annotation file.

        Returns:
            list[dict]: Per-sample data dicts containing ``img_path``,
                ``img_path2``, ``seg_map_path``, ``question`` (raw string),
                ``answer`` (int), ``label_map``, ``reduce_zero_label``, and
                ``seg_fields``.
        """
        assert osp.isfile(self.ann_file), \
            f'Annotation file not found: {self.ann_file}'

        with open(self.ann_file, 'r') as f:
            content = f.read().strip()

        # Support both JSON array and JSON-Lines formats
        if content.startswith('['):
            records = json.loads(content)
        else:
            records = [json.loads(line) for line in content.splitlines()
                       if line.strip()]

        img_prefix = self.data_prefix.get('img_path', '')
        img_prefix2 = self.data_prefix.get('img_path2', '')
        ann_prefix = self.data_prefix.get('seg_map_path', '')

        data_list = []
        for rec in records:
            img_path = self._resolve_path(
                img_prefix, rec['img_path'])
            img_path2 = self._resolve_path(
                img_prefix2, rec['img_path2'])
            seg_map_path = self._resolve_path(
                ann_prefix, rec['seg_map_path'])

            question_text = rec.get('question', '')
            answer = int(rec.get('answer', 0))

            data_info = dict(
                img_path=img_path,
                img_path2=img_path2,
                seg_map_path=seg_map_path,
                question=question_text,
                answer=answer,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[])
            data_list.append(data_info)

        return data_list
