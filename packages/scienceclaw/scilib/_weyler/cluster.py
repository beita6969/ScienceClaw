from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, Tuple, Union
import numpy as np
import torch

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

class PrecisionMatrix():
  """ Contains parameters of precision matrix and computes probability margins.
  """
  def __init__(self, mode: str, values: Union[np.ndarray, torch.Tensor], sigma_scale: float, alpha_scale: float, height: float):
    self.mode = mode # {'np', 'torch'}
    self.alpha_scale = alpha_scale
    self.sigma_scale = sigma_scale
    self.height = height
    self.alpha = 0 # rotation

    if self.mode == 'np':
      l_11 = np.exp(values[0] * self.sigma_scale)

      try:
        l_22 = np.exp(values[1] * self.sigma_scale)
      except IndexError:
        l_22 = l_11

      try:
        self.alpha = (values[2] * self.alpha_scale) * (math.pi / 2)
      except IndexError:
        self.alpha = np.zeros_like(l_11)

      self.p_xx = np.power(np.cos(self.alpha), 2) *  l_11 + np.power(np.sin(self.alpha), 2) * l_22
      self.p_yy = np.power(np.sin(self.alpha), 2) *  l_11 + np.power(np.cos(self.alpha), 2) * l_22
      self.p_yx = -np.cos(self.alpha) * np.sin(self.alpha) * l_11 + np.sin(self.alpha)*np.cos(self.alpha) * l_22
      self.p_xy = self.p_yx

      # precision matrix
      self.precision_mat = np.array([[self.p_xx, self.p_xy], [self.p_yx, self.p_yy]])

      # positive defintie check
      eigvals, _ = np.linalg.eig(self.precision_mat)
      assert np.all((eigvals + 1e-8) >= 0)

    elif self.mode == 'torch':
      l_11 = torch.exp(values[0] * self.sigma_scale)

      try:
        l_22 = torch.exp(values[1] * self.sigma_scale)
      except IndexError:
        l_22 = l_11

      try:
        self.alpha = torch.tanh(values[2] * self.alpha_scale) * (math.pi / 2)
      except IndexError:
        self.alpha = torch.zeros_like(l_11)

      _p_xx = torch.pow(l_11, 2)
      _p_yy = torch.pow(l_22, 2)

      self.p_xx = torch.pow(torch.cos(self.alpha), 2) * _p_xx + torch.pow(torch.sin(self.alpha), 2) * _p_yy
      self.p_yy = torch.pow(torch.sin(self.alpha), 2) * _p_xx + torch.pow(torch.cos(self.alpha), 2) * _p_yy
      self.p_yx = -torch.cos(self.alpha)*torch.sin(self.alpha)*_p_xx + torch.sin(self.alpha)*torch.cos(self.alpha)* _p_yy
      self.p_xy = self.p_yx

      # precision matrix
      if values.is_cuda:
        self.precision_mat = torch.cuda.FloatTensor([[self.p_xx, self.p_yx], [self.p_yx, self.p_yy]])
      else:
        self.precision_mat = torch.Tensor([[self.p_xx, self.p_xy], [self.p_yx, self.p_yy]])

      # positive defintie check
      eigvals = torch.eig(self.precision_mat)
      assert all(eigvals.eigenvalues[:,0] >= 0)

  def gaussian(self, dx: float, dy: float) -> float:
    """ Compute 2D gaussian

    Args:
        dx (float): offset from mean in x-direction
        dy (float): offset from mean in y-direction

    Returns:
        float: value of 2D Gaussian
    """
    if self.mode == 'np':
      value = np.exp(-(1/2)* ((dx * self.p_xx * dx) + (dy * self.p_yx * dx) +  (dx * self.p_yx * dy) + (dy * self.p_yy * dy)))
    elif self.mode == 'torch':
      value = torch.exp(-(1/2)* ((dx * self.p_xx * dx) + (dy * self.p_yx * dx) +  (dx * self.p_yx * dy) + (dy * self.p_yy * dy)))

    return value

  def gaussian_margin(self, x: np.ndarray, p: float=0.5) -> float:
    """ Compute the margin for a given probability.

    Args:
        x (np.ndarray): vector of length 1 of shape (2,)
        p (float, optional): probability margin. Defaults to 0.5.

    Returns:
        float: scale of vector x to reach the margin with probability p
    """
    if self.mode == 'np':
      margin = np.sqrt(- (2*np.log(p)) / (x[0]**2*self.p_xx + x[1]*self.p_yx*x[0] + x[0]*self.p_yx*x[1] + x[1]**2*self.p_yy))
    elif self.mode == 'torch':
      margin = torch.sqrt(- (2*torch.log(p)) / (x[0]**2*self.p_xx + x[1]*self.p_yx*x[0] + x[0]*self.p_yx*x[1] + x[1]**2*self.p_yy))

    x_margin = x[0] * margin
    y_margin = x[1] * margin
    check_margin = self.gaussian(x_margin, y_margin)
    if self.mode == 'np':
      assert np.abs(check_margin - p) < 1e-2
    elif self.mode == 'torch':
      assert torch.abs(check_margin - p) < 1e-2

    return margin

  def get_ellipse_params(self) -> Tuple[int, int, float, np.ndarray, np.ndarray]:
    """ Compute the ellipse param of this precision matrix

    Returns:
        Tuple[int, int, float]: height, width and angle
    """
    if not isinstance(self.precision_mat, np.ndarray):
      self.precision_mat = self.precision_mat.cpu().numpy()

    eigvals, eigvecs = np.linalg.eig(self.precision_mat)
    assert np.all(eigvals >= 0)

    # compute major and minor axis
    width = int(np.ceil(self.gaussian_margin(eigvecs[:,0]) * self.height))
    height = int(np.ceil(self.gaussian_margin(eigvecs[:,1]) * self.height))
    angle = np.arctan2(eigvecs[1,0], eigvecs[0,0])
    angle = np.arctan2(np.sin(self.alpha), np.cos(self.alpha))

    v1 = eigvecs[:,0] * width
    v2 = eigvecs[:,1] * height

    # HACK to compute width and height correctly
    r = np.array([[np.cos(self.alpha), -np.sin(self.alpha)], [np.sin(self.alpha), np.cos(self.alpha)]])
    save_pxx = self.p_xx
    save_pyy = self.p_yy
    save_pyx = self.p_yx
    save_pxy = self.p_xy
    pmat = r @ self.precision_mat @ r.T # transform basis
    self.p_xx = pmat[0,0]
    self.p_yy = pmat[1,1]
    self.p_yx = pmat[1,0]
    self.p_xy = pmat[0,1]
    width = int(np.ceil(self.gaussian_margin(np.array([1, 0])) * self.height)) # := scale of x-axis
    height = int(np.ceil(self.gaussian_margin(np.array([0, 1])) * self.height)) # := scale of y-axis
    self.p_xx = save_pxx
    self.p_yy = save_pyy
    self.p_yx = save_pyx
    self.p_xy = save_pxy

    return (height, width , angle, v1, v2)

class Cluster():
  """ Postprocessing to cluster parts and objects based on spatial embeddings.
  """
  def __init__(self,
               mode: str,
               width: int,
               height: int,
               n_classes:int,
               n_sigma: int,
               sigma_scale:float,
               alpha_scale: float,
               parts_area_thres: int,
               parts_score_thres: float,
               objects_area_thres: int,
               objects_score_thres: float,
               apply_offsets: bool = True):
    self.mode = mode # {'np', 'torch'}
    self.width = width
    self.height = height
    self.n_classes = n_classes
    self.n_sigma = n_sigma
    self.sigma_scale = sigma_scale
    self.alpha_scale = alpha_scale
    self.parts_area_thres = parts_area_thres
    self.parts_score_thres = parts_score_thres
    self.objects_area_thres = objects_area_thres
    self.objects_score_thres = objects_score_thres
    self.apply_offsets = apply_offsets

    # build coordinate map
    x_max = int(round(self.width / self.height))
    if self.mode == 'np':
      xm = np.linspace(0, x_max, self.width).reshape(1, 1, -1).repeat(self.height, axis=1)
      ym = np.linspace(0, 1, self.height).reshape(1, -1, 1).repeat(self.width, axis=2)
      self.xym = np.concatenate((xm, ym), axis=0) # (2 x h x w)
    elif self.mode == 'torch':
      xm = torch.linspace(0, x_max, self.width).view(1, 1, -1).expand(1, self.height, self.width)
      ym = torch.linspace(0, 1, self.height).view(1, -1, 1).expand(1, self.height, self.width)
      self.xym = torch.cat((xm, ym), 0) # (2 x h x w)

      if torch.cuda.is_available():
        self.xym = self.xym.cuda()
    else:
      assert False, 'Cluster mode is not valid - set it either to "np" or "torch".'

  def cluster(self, pred: Union[np.ndarray, torch.Tensor]) -> Dict:
    """ Perform clustering based on network predictions.

    Args:
        pred (Union[np.ndarray, torch.Tensor]): network predicition (chans, h, w)

        The instance's mode should be set 'np' or 'torch' accordingly.

    Returns:
        Dict: results of clustering
    """
    height, width = pred.shape[1], pred.shape[2]
    xym = self.xym[:, :height, :width]

    # compute indicies
    start_objects_spatial_emb = 0
    end_objects_spatial_emb = 2

    start_objects_sigma = end_objects_spatial_emb
    end_objects_sigma = start_objects_sigma + self.n_sigma

    start_parts_spatial_emb = end_objects_sigma
    end_parts_spatial_emb = start_parts_spatial_emb + 2

    start_parts_sigma = end_parts_spatial_emb
    end_parts_sigma = start_parts_sigma + self.n_sigma

    start_objects_seed = end_parts_sigma
    end_objects_seed = start_objects_seed + self.n_classes

    start_parts_seed = end_objects_seed
    end_parts_seed = start_parts_seed + self.n_classes

    # extract predictions
    if self.mode == 'np':
      parts_offsets = np.tanh(pred[start_parts_spatial_emb: end_parts_spatial_emb]) # (2, h, w)
      objects_offsets = np.tanh(pred[start_objects_spatial_emb: end_objects_spatial_emb]) # (2, h, w)

      parts_seed = sigmoid(pred[start_parts_seed: end_parts_seed]) # (n_classes, h, w)
      objects_seed = sigmoid(pred[start_objects_seed: end_objects_seed]) # (n_classes, h, w)
    elif self.mode == 'torch':
      parts_offsets = torch.tanh(pred[start_parts_spatial_emb: end_parts_spatial_emb]) # (2, h, w)
      objects_offsets = torch.tanh(pred[start_objects_spatial_emb: end_objects_spatial_emb]) # (2, h, w)

      parts_seed = torch.sigmoid(pred[start_parts_seed: end_parts_seed]) # (n_classes, h, w)
      objects_seed = torch.sigmoid(pred[start_objects_seed: end_objects_seed]) # (n_classes, h, w)

    if self.apply_offsets:
      parts_spatial_emb = xym + parts_offsets # (2, h, w)
      objects_spatial_emb = xym + parts_offsets + objects_offsets # (2, h, w)
    else:
      parts_spatial_emb = xym
      objects_spatial_emb = xym

    parts_sigma = pred[start_parts_sigma:end_parts_sigma] # (n_sigma, h, w)
    objects_sigma = pred[start_objects_sigma: end_objects_sigma] # (n_sigma, h, w)

    # --- clustering ---
    results = {}
    results["parts"] = defaultdict(list)
    results["objects"] = defaultdict(list)

    object_count = 0
    part_count = 0

    # start clustering of parts
    for cls_idx in range(self.n_classes):
      # get all foreground pixels
      parts_cls_mask = parts_seed[cls_idx] > 0.5 # (h, w)

      if parts_cls_mask.sum() > self.parts_area_thres:
        if self.mode == 'np':
          # get coords of pixels which belong to foreground
          part_xym_masked = xym[np.broadcast_to(parts_cls_mask, xym.shape)].reshape(2, -1) # (2, n)

          # get all predictions which belong to foreground
          parts_spatial_emb_masked = parts_spatial_emb[np.broadcast_to(parts_cls_mask, parts_spatial_emb.shape)].reshape(2, -1) # (2, n)
          objects_spatial_emb_masked = objects_spatial_emb[np.broadcast_to(parts_cls_mask, objects_spatial_emb.shape)].reshape(2, -1) # (2, n)
          parts_sigma_masked = parts_sigma[np.broadcast_to(parts_cls_mask, parts_sigma.shape)].reshape(self.n_sigma, -1) # (n_sigma, n)
          parts_seed_masked = parts_seed[cls_idx][parts_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          parts_unclustered = np.ones(parts_cls_mask.sum(), dtype=np.bool_) # (n, )
        elif self.mode == 'torch':
          part_xym_masked = xym[parts_cls_mask.expand_as(xym)].reshape(2, -1) # (2, n)

          parts_spatial_emb_masked = parts_spatial_emb[parts_cls_mask.expand_as(parts_spatial_emb)].reshape(2, -1) # (2, n)
          objects_spatial_emb_masked = objects_spatial_emb[parts_cls_mask.expand_as(objects_spatial_emb)].reshape(2, -1) # (2, n)
          parts_sigma_masked = parts_sigma[parts_cls_mask.expand_as(parts_sigma)].reshape(self.n_sigma, -1) # (n_sigma, n)
          parts_seed_masked = parts_seed[cls_idx][parts_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          parts_unclustered = torch.ones(parts_cls_mask.sum()).byte() # (n, )
          if torch.cuda.is_available():
            parts_unclustered = parts_unclustered.cuda()

        # cluster each part
        while(parts_unclustered.sum()) > self.parts_area_thres:
          if self.mode == 'np':
            # get pixel with highest score which is still unclustered
            idx = (parts_seed_masked * parts_unclustered.astype(np.float32)).argmax().item()
          elif self.mode == 'torch':
            idx = (parts_seed_masked * parts_unclustered.float()).argmax().item()

          part_seed_score = parts_seed_masked[idx]
          if part_seed_score < self.parts_score_thres:
            break

          part_center = parts_spatial_emb_masked[: , idx] # (2, )
          part_sigmas = parts_sigma_masked[:, idx] # (n_sigma)
          part_prec_mat = PrecisionMatrix(self.mode, part_sigmas, self.sigma_scale, self.alpha_scale, height)

          # calculate gaussian distance between center to all foreground pixels
          delta = (parts_spatial_emb_masked - part_center.reshape(2, 1)) # (2, n)
          if self.mode == 'np':
            prob = np.exp(-(1/2) * (
                                    (delta[0] * part_prec_mat.p_xx * delta[0]) + \
                                    (delta[1] * part_prec_mat.p_yx * delta[0]) + \
                                    (delta[0] * part_prec_mat.p_xy * delta[1]) + \
                                    (delta[1] * part_prec_mat.p_yy * delta[1])
                                   )
                         ) # (n, )
          elif self.mode == 'torch':
            prob = torch.exp(-(1/2) * ((delta[0] * part_prec_mat.p_xx * delta[0]) + \
                                       (delta[1] * part_prec_mat.p_yx * delta[0]) + \
                                       (delta[0] * part_prec_mat.p_yx * delta[1]) + \
                                       (delta[1] * part_prec_mat.p_yy * delta[1])
                                      )) # (n, )

          part_proposal = ((prob * parts_unclustered) > 0.5)
          parts_unclustered[idx] = False # set center as clustered to avoid never ending loop

          if part_proposal.sum() > self.parts_area_thres:
            parts_unclustered[part_proposal] = False
            # binary mask for this part
            part_instance_map = np.zeros((height, width), dtype=np.bool_)
            if self.mode == 'np':
              part_instance_map[parts_cls_mask] = part_proposal

              # spatial embeddings of this part
              part_spatial_emb = parts_spatial_emb_masked[np.broadcast_to(part_proposal, parts_spatial_emb_masked.shape)].reshape(2, -1)

              # spatial embeddings of this part w.r.t to object
              object_spatial_emb = objects_spatial_emb_masked[np.broadcast_to(part_proposal, objects_spatial_emb_masked.shape)].reshape(2, -1)

              # coordinates of this part in original image
              part_xym = part_xym_masked[np.broadcast_to(part_proposal, part_xym_masked.shape)].reshape(2, -1)
            elif self.mode == 'torch':
              if not isinstance(parts_cls_mask, np.ndarray):
                parts_cls_mask = parts_cls_mask.cpu().numpy().astype(np.bool_)
              part_instance_map[parts_cls_mask] = part_proposal.cpu().numpy()

               # spatial embeddings of this part
              part_spatial_emb = parts_spatial_emb_masked[part_proposal.expand_as(parts_spatial_emb_masked)].reshape(2, -1).cpu().numpy()

              # spatial embeddings of this part w.r.t to object
              object_spatial_emb = objects_spatial_emb_masked[part_proposal.expand_as(objects_spatial_emb_masked)].reshape(2, -1).cpu().numpy()

              # coordinates of this part in original image
              part_xym = part_xym_masked[part_proposal.expand_as(part_xym_masked)].reshape(2, -1).cpu().numpy()

            part_count += 1
            part = {"part_mask": part_instance_map,
                    "part_embeddings": part_spatial_emb,
                    "object_embeddings": object_spatial_emb,
                    "part_coordinates": part_xym,
                    "part_score": part_seed_score,
                    "part_center": part_center,
                    "part_sigma": part_prec_mat,
                    "part_id": part_count,
                    "belongs_to_object": False}

            results["parts"][str(cls_idx)].append(part)

      # --- finished clustering parts of current cls - now let's merge them to objects ---
      # get all foreground pixels
      objects_cls_mask = objects_seed[cls_idx] > 0.5 # (h, w)
      if objects_cls_mask.sum() > self.objects_area_thres:
        if self.mode == 'np':
          # get coords of pixels which belong to (object) foreground
          objects_xym_masked = xym[np.broadcast_to(objects_cls_mask, xym.shape)].reshape(2, -1) # (2, n)

          # get all predictions which belong to (object) foreground
          objects_spatial_emb_masked = objects_spatial_emb[np.broadcast_to(objects_cls_mask, objects_spatial_emb.shape)].reshape(2, -1) # (2, n)
          objects_sigma_masked = objects_sigma[np.broadcast_to(objects_cls_mask, objects_sigma.shape)].reshape(self.n_sigma, -1) # (n_sigma, n)
          objects_seed_masked = objects_seed[cls_idx][objects_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          objects_unclustered = np.ones(objects_cls_mask.sum(), dtype=np.bool_) # (n, )
        elif self.mode == 'torch':
          objects_xym_masked = xym[objects_cls_mask.expand_as(xym)].reshape(2, -1) # (2, n)

          objects_spatial_emb_masked = objects_spatial_emb[objects_cls_mask.expand_as(objects_spatial_emb)].reshape(2, -1) # (2, n)
          objects_sigma_masked = objects_sigma[objects_cls_mask.expand_as(objects_sigma)].reshape(self.n_sigma, -1) # (n_sigma, n)
          objects_seed_masked = objects_seed[cls_idx][objects_cls_mask] # (n, )

          # set all foreground pixels to be unclustered
          objects_unclustered = torch.ones(objects_cls_mask.sum()).byte() # (n, )
          if torch.cuda.is_available():
            objects_unclustered = objects_unclustered.cuda()

        while(objects_unclustered.sum()) > self.objects_area_thres:
          if self.mode == 'np':
            idx = (objects_seed_masked * objects_unclustered.astype(np.float32)).argmax().item()
          elif self.mode == 'torch':
            idx = (objects_seed_masked * objects_unclustered.float()).argmax().item()

          objects_seed_score = objects_seed_masked[idx]
          if objects_seed_score < self.objects_score_thres:
            break

          object_center = objects_spatial_emb_masked[:, idx]  # (2, )
          object_sigmas = objects_sigma_masked[:, idx] # (n_sigma)
          object_prec_mat = PrecisionMatrix(self.mode, object_sigmas, self.sigma_scale, self.alpha_scale, height)
          objects_unclustered[idx] = False

          object_has_parts = False
          object_area = 0.0

          # assign each part of same cls to an object
          object_part_idx = []
          for idx, part in enumerate(results["parts"][str(cls_idx)]):
            if part["belongs_to_object"]:
              # do not assign any object twice or more
              continue

            # calculate gaussian
            if self.mode == 'np':
              delta = part["object_embeddings"] - object_center.reshape(2, 1)
              prob = np.exp(-(1/2) * ((delta[0] * object_prec_mat.p_xx * delta[0]) + \
                                      (delta[1] * object_prec_mat.p_yx * delta[0]) + \
                                      (delta[0] * object_prec_mat.p_yx * delta[1]) + \
                                      (delta[1] * object_prec_mat.p_yy * delta[1])
                                     )) # (n, )
              # check if at least 50 percent of part embeddings are within the object
              ratio = (np.sum(prob > 0.5)) / (prob.shape[0])
            elif self.mode == 'torch':
              part_object_embeddings = torch.from_numpy(part["object_embeddings"])
              if torch.cuda.is_available:
                part_object_embeddings = part_object_embeddings.cuda()

              delta = part_object_embeddings - object_center.reshape(2, 1)
              prob = torch.exp(-(1/2) * ((delta[0] * object_prec_mat.p_xx * delta[0]) + \
                                         (delta[1] * object_prec_mat.p_yx * delta[0]) + \
                                         (delta[0] * object_prec_mat.p_yx * delta[1]) + \
                                         (delta[1] * object_prec_mat.p_yy * delta[1])
                                        )) # (n, )
              # check if at least 50 percent of part embeddings are within the object
              ratio = (torch.sum(prob > 0.5)).float() / float((prob.shape[0]))

            if ratio < 0.5:
              # the amount of embeddings within this cluster is not sufficient
              continue

            object_has_parts = True
            object_area += part["part_mask"].sum()
            object_part_idx.append(idx)
            part["belongs_to_object"] = True

            # set pixels which belong to current part to be clustered
            for j in range(part["part_coordinates"].shape[1]):
              part_coord = part["part_coordinates"][:, j].reshape(2, 1)
              if self.mode == 'np':
                # check if this coordinate is in the object foreground mask
                part_coord_in_fg_mask = np.all(objects_xym_masked == part_coord, axis=0)
                if np.any(part_coord_in_fg_mask):
                  objects_unclustered[part_coord_in_fg_mask.argmax()] = False
              elif self.mode == 'torch':
                part_coord = torch.Tensor(part_coord)
                part_coord = part_coord.to(objects_xym_masked.device)
                part_coord_in_fg_mask = (objects_xym_masked == part_coord).all(dim=0)
                if part_coord_in_fg_mask.any():
                  objects_unclustered[part_coord_in_fg_mask.argmax()] = False

          if object_has_parts:
            if object_area > self.objects_area_thres:
              object_count += 1
              if self.mode == 'np':
                obj = {"obj_part_indicies": object_part_idx,
                       "obj_score": objects_seed_score,
                       "obj_center": object_center,
                       "obj_sigma": object_prec_mat,
                       "object_id": object_count}
              elif self.mode == 'torch':
                obj = {"obj_part_indicies": object_part_idx,
                       "obj_score": objects_seed_score.cpu().numpy(),
                       "obj_center": object_center.cpu().numpy(),
                       "obj_sigma": object_prec_mat,
                       "object_id": object_count}

              results["objects"][str(cls_idx)].append(obj)


    return objects_seed, parts_seed, objects_offsets, parts_offsets, objects_sigma, parts_sigma, results
