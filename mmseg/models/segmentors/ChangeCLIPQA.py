# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import SampleList

from .ChangeCLIPCD import ChangeCLIP


@MODELS.register_module()
class ChangeCLIPQA(ChangeCLIP):
    """ChangeCLIP model extended for Question Answering (QA) tasks in remote
    sensing change detection.

    Extends ChangeCLIP with:
    - ``answer_head``: Predicts answers to 23-class questions based on combined
      text and vision embeddings.
    - ``q_vis_cross_attn``: Cross-attention fusion between question embeddings
      and vision features.
    - Updated loss: ``total_loss = CE(mask) + 0.2 * CE(answer)``.

    Inputs include ``img1``, ``img2`` (as a concatenated 6-channel tensor) and
    ``question`` text tokens stored in ``data_samples``.

    The question text is tokenised with ``context_length`` (model-level, e.g.
    64) and encoded through the shared ``text_encoder`` using ``contexts2``,
    mirroring the way change-caption texts are encoded in the base
    ``ChangeCLIP``.

    Args:
        num_answer_classes (int): Number of answer classes for QA head.
            Defaults to 23.
        answer_head_channels (int): Intermediate channel size for the answer
            head MLP. Defaults to 512.
        answer_loss_weight (float): Weight for the answer classification loss.
            Defaults to 0.2.
        cross_attn_heads (int): Number of attention heads for
            ``q_vis_cross_attn``. Defaults to 8.
        vis_proj_channels (int): Channel dimension of multi-scale vision
            features fed into cross-attention (output dim of FPN).
            Defaults to 256.
        text_embed_dim (int): Dimension of the text/question embeddings used
            by the cross-attention and answer head. Defaults to 1024.
        **kwargs: Additional keyword arguments forwarded to ``ChangeCLIP``.
    """

    def __init__(self,
                 num_answer_classes: int = 23,
                 answer_head_channels: int = 512,
                 answer_loss_weight: float = 0.2,
                 cross_attn_heads: int = 8,
                 vis_proj_channels: int = 256,
                 text_embed_dim: int = 1024,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_answer_classes = num_answer_classes
        self.answer_loss_weight = answer_loss_weight
        self.text_embed_dim = text_embed_dim

        # Project vision features to text_embed_dim for cross-attention
        self.vis_proj = nn.Linear(vis_proj_channels, text_embed_dim)

        # Cross-attention between question embeddings and vision features
        self.q_vis_cross_attn = nn.MultiheadAttention(
            embed_dim=text_embed_dim,
            num_heads=cross_attn_heads,
            batch_first=True)

        # Answer prediction head: MLP over fused question-vision features
        self.answer_head = nn.Sequential(
            nn.Linear(text_embed_dim, answer_head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(answer_head_channels, num_answer_classes))

        # Loss for answer classification
        self.answer_criterion = nn.CrossEntropyLoss()

    def _get_question_tokens(self, data_samples: SampleList) -> Tensor:
        """Extract and tokenise question text from data samples.

        Each data sample is expected to carry a ``question`` field (set via
        ``PackCDInputs`` meta_keys) containing a raw question string.  The
        strings are tokenised using the model's ``context_length`` and returned
        as a batch tensor of shape ``(B, 1, context_length)`` matching the
        ``CLIPTextContextEncoder`` input format ``(N, K, context_length)``.

        Args:
            data_samples (SampleList): Batch of data samples.

        Returns:
            Tensor: Question tokens of shape ``(B, 1, context_length)``.
        """
        from ..utils.untils import tokenize as clip_tokenize

        tokens = []
        for ds in data_samples:
            # Support both attribute access (after set_metainfo) and
            # dict-style access.
            if hasattr(ds, 'question'):
                q = ds.question
            elif hasattr(ds, 'metainfo') and 'question' in ds.metainfo:
                q = ds.metainfo['question']
            else:
                q = ds['question']
            # q is a raw string; tokenise with the model's context_length.
            # Result: (1, context_length)
            tok = clip_tokenize(q, context_length=self.context_length)
            tokens.append(tok)
        # Stack to (B, 1, context_length)
        return torch.cat(tokens, dim=0).unsqueeze(1)

    def _get_answer_labels(self, data_samples: SampleList) -> Tensor:
        """Extract answer class labels from data samples.

        Each data sample is expected to carry an ``answer`` field (set via
        ``PackCDInputs`` meta_keys) containing an integer class index.

        Args:
            data_samples (SampleList): Batch of data samples.

        Returns:
            Tensor: Answer labels of shape ``(B,)``.
        """
        labels = []
        for ds in data_samples:
            if hasattr(ds, 'answer'):
                a = ds.answer
            elif hasattr(ds, 'metainfo') and 'answer' in ds.metainfo:
                a = ds.metainfo['answer']
            else:
                a = ds['answer']
            labels.append(int(a))
        return torch.tensor(labels, dtype=torch.long)

    def _encode_question(self, question_tokens: Tensor) -> Tensor:
        """Encode question tokens into embeddings using the shared text encoder.

        Uses the same ``contexts2`` learnable context as the change-text
        pathway so that the question and change-text embeddings live in the
        same feature space.

        Args:
            question_tokens (Tensor): Shape ``(B, 1, context_length)``.

        Returns:
            Tensor: Question embeddings of shape ``(B, 1, text_embed_dim)``.
        """
        B = question_tokens.shape[0]
        device = question_tokens.device
        # contexts2: (1, 1, N2, token_embed_dim) → (B, 1, N2, token_embed_dim)
        contexts_ = torch.cat([self.contexts2] * B, dim=0).to(device)
        # text_encoder forward: text (B, K=1, ctx_len) → (B, 1, embed_dim)
        q_embeddings = self.text_encoder(question_tokens, contexts_)
        return q_embeddings

    def _predict_answer(self,
                        q_embeddings: Tensor,
                        vis_features: List[Tensor]) -> Tensor:
        """Predict answer class by fusing question embeddings with vision
        features via cross-attention.

        The last (most semantic) feature scale is used as visual context.
        Vision features are projected to ``text_embed_dim`` before
        cross-attention.

        Args:
            q_embeddings (Tensor): Question embeddings of shape
                ``(B, 1, text_embed_dim)``.
            vis_features (List[Tensor]): Multi-scale vision feature maps, each
                of shape ``(B, C_vis, H, W)``.

        Returns:
            Tensor: Answer logits of shape ``(B, num_answer_classes)``.
        """
        # Use the last (most semantic) feature scale as visual context
        vis = vis_features[-1]  # (B, C_vis, H, W)
        B, C_vis, H, W = vis.shape

        # Flatten spatial dims and project to text_embed_dim
        vis_flat = vis.flatten(2).permute(0, 2, 1)  # (B, H*W, C_vis)
        vis_flat = self.vis_proj(vis_flat)            # (B, H*W, text_embed_dim)

        # Cross-attention: query=question, key/value=vision features
        attn_out, _ = self.q_vis_cross_attn(
            query=q_embeddings,
            key=vis_flat,
            value=vis_flat)  # (B, 1, text_embed_dim)

        # Squeeze token dim and classify
        fused = attn_out.squeeze(1)              # (B, text_embed_dim)
        answer_logits = self.answer_head(fused)  # (B, num_answer_classes)
        return answer_logits

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Computes both the segmentation mask loss (via the decode head) and the
        answer classification loss, combining them as:
        ``total_loss = CE(mask) + answer_loss_weight * CE(answer)``.

        Args:
            inputs (Tensor): Concatenated image pair of shape
                ``(B, 6, H, W)``.
            data_samples (SampleList): Batch of data samples containing
                ``gt_sem_seg``, ``question`` and ``answer`` fields.

        Returns:
            dict[str, Tensor]: Dictionary of loss components.
        """
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        textA, textB = self.get_cls_text(data_samples)
        text_embeddingsA, x_clipA, score_mapA = self.after_extract_feat_clip(
            xA, textA)
        text_embeddingsB, x_clipB, score_mapB = self.after_extract_feat_clip(
            xB, textB)

        x_orig = [
            torch.cat([x_clipA[i], x_clipB[i]], dim=1)
            for i in range(len(x_clipA))]
        x_minus = [
            self.minus_conv[i](torch.abs(x_clipA[i] - x_clipB[i]))
            for i in range(len(x_clipA))]
        x_diff = [
            F.sigmoid(
                1 - torch.cosine_similarity(x_clipA[i], x_clipB[i], dim=1)
            ).unsqueeze(1)
            for i in range(len(x_clipA))]

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        if self.text_head:
            x = [text_embeddingsA] + x_orig
        else:
            x = x_orig

        x = [
            torch.cat([x[i] * x_diff[i], x_minus[i], x[i]], dim=1)
            for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        losses = dict()

        # Segmentation mask loss
        loss_decode = self._decode_head_forward_train_with_text(
            x, text_embeddingsA, text_embeddingsB, data_samples)
        losses.update(loss_decode)

        # Answer classification loss
        question_tokens = self._get_question_tokens(data_samples).to(
            inputs.device)
        q_embeddings = self._encode_question(question_tokens)
        answer_logits = self._predict_answer(q_embeddings, x)
        answer_labels = self._get_answer_labels(data_samples).to(
            answer_logits.device)
        loss_answer = self.answer_criterion(answer_logits, answer_labels)
        losses['loss_answer'] = self.answer_loss_weight * loss_answer

        return losses

