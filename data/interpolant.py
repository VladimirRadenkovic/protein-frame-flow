import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from scipy.optimize import linear_sum_assignment

import numpy as np

KB = 0.83144626181
T = 300
KBT = KB*T
MASS = 12.011

INERTIA =  ((26.02177038, 10.02305903, 0.),
              (10.02305903, 31.83040681, 0.),
              (0., 0., 57.85217719)
        )
I = torch.tensor(INERTIA)
I_inv = torch.inverse(I)

eps = 0.01
t = 1 - eps
g_sq = 2*(1-t)/(t)
#m = 1/t
#one_minus_t_inv = 1/(1 - t)
dt = 2e-3 #2fs (units in ps)
#dt = 1
#dt = 0.001 
#A = 7*1e-5
#A = 2*1e-4
A = 1.4*1e-4
#A = 2*1e-2
#A = 4.5*1e-4
#A = 4*1e-4

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
       t = torch.rand(num_batch, device=self._device)
       return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, motif_mask = None):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        # center T_0^s
        mean = (trans_1*(1-motif_mask[..., None])).mean(dim=1, keepdim=True).to(trans_0.device)
        trans_0 = trans_0 - mean

        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        trans_t = trans_t * motif_mask[..., None] + (trans_t - mean) * (1 - motif_mask[..., None])
        return trans_t * res_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)
    

    def _motif_scaffold_partition(self, num_residues, min_percent, max_percent, device):
        s = torch.randint(int(num_residues * min_percent), int(num_residues * max_percent) + 1, (1,)).item()
        m = torch.randint(1, s + 1, (1,)).item()

        motif_indices = set()
        for i in range(1, m + 1):
            available_indices = list(set(range(num_residues)) - motif_indices)
            start_location = available_indices[torch.randint(0, len(available_indices), (1,)).item()]
            max_possible_length = s - m + i - len(motif_indices)
            motif_length = torch.randint(1, max_possible_length + 1, (1,)).item()
            end_location = min(start_location + motif_length, num_residues)
            motif_indices.update(range(start_location, end_location))

        motif_indices = sorted(list(motif_indices))
        mask = torch.ones(num_residues, dtype=torch.int, device=device)
        mask[list(motif_indices)] = 0

        return mask, motif_indices
    


    
    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']
        # [B, N]
        res_mask = batch['res_mask']
        num_batch, _ = res_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def corrupt_batch_with_motif_amortization(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        res_mask = batch['res_mask']
        motif_mask = batch['motif_mask']
        
        


        #res_mask = motif_scaffold_partition(noisy_batch['res_mask'].shape[1], min_percent = 0.05, max_percent = 0.5, device = self._device)
        """
        all_motif_masks = []
        for _ in range(len(noisy_batch['res_mask'])):
            motiff_mask, _ = self._motif_scaffold_partition(noisy_batch['res_mask'].shape[1], min_percent = 0.05, max_percent = 0.5, device = self._device)
            all_motif_masks.append(motiff_mask)
        motif_masks = torch.cat(all_motif_masks, dim=0)
        res_mask = batch['res_mask']
        """
        num_batch, _ = res_mask.shape

        t = self.sample_t(num_batch)[:, None]
        noisy_batch['t'] = t

        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, t, res_mask, motif_mask)
        trans_t = _trans_diffuse_mask(trans_t, trans_1, motif_mask)

        #mean = (trans_t*(1-motif_mask[..., None])).mean(dim=1, keepdim=True).to(trans_t.device)
        #trans_t = trans_t * motif_mask[..., None] + (trans_t - mean) * (1 - motif_mask[..., None])

        #motif_mask_exp = motif_mask.squeeze(0)
        #trans_t[motif_mask_exp == 0] = trans_t[motif_mask_exp == 0] - trans_t[motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_t.device)

        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, motif_mask)
        noisy_batch['rotmats_t'] = rotmats_t


        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t
    

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)
    

    def sample(
            self,
            num_batch,
            num_res,
            model,
        ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        batch = {
            'res_mask': res_mask,
            'motif_mask': res_mask,
        }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        #for t_2 in range(3):
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj


    def sample_amortized(
            self,
            batch,
            model,
        ):
        num_batch, num_res = batch['res_mask'].shape
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        motif_mask = batch['motif_mask']
        batch['res_mask'] = res_mask
        trans_1 = batch['trans_1']
        rotmats_1 = batch['rotmats_1']

        motif_mask_exp = motif_mask.squeeze(0)
        trans_m = trans_1[:,motif_mask_exp == 0] - trans_1[:,motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_1.device)
        

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        trans_0[:,motif_mask_exp == 1] = trans_0[:,motif_mask_exp == 1] -  trans_1[:,motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_1.device)
        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]
        prot_traj = [(trans_0, rotmats_0)]
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_traj = []
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]

            trans_t_1 = _trans_diffuse_mask(trans_t_1, trans_1, motif_mask)
            trans_t_1[:,motif_mask_exp == 0] = trans_m
            rotmats_t_1 = _rots_diffuse_mask(rotmats_t_1, rotmats_1, motif_mask)

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        trans_t_1 = _trans_diffuse_mask(trans_t_1, trans_1, motif_mask)
        trans_t_1[:, motif_mask_exp == 0] = trans_m

        rotmats_t_1 = _rots_diffuse_mask(rotmats_t_1, rotmats_1, motif_mask)
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_1, motif_mask)
        pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_1, motif_mask)

        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj
    
    """
    def _sample_init(
            self,
            batch,
            model,
        ):
        num_batch, num_res = batch['res_mask'].shape
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        motif_mask = batch['motif_mask']
        batch['res_mask'] = res_mask
        trans_1 = batch['trans_1']
        rotmats_1 = batch['rotmats_1']

        motif_mask_exp = motif_mask.squeeze(0)
        trans_m = trans_1[:,motif_mask_exp == 0] - trans_1[:,motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_1.device)
        

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        trans_0[:,motif_mask_exp == 1] = trans_0[:,motif_mask_exp == 1] -  trans_1[:,motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_1.device)
        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]
        prot_traj = (trans_0, rotmats_0)
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj

            trans_t_1 = _trans_diffuse_mask(trans_t_1, trans_1, motif_mask)
            trans_t_1[:,motif_mask_exp == 0] = trans_m
            rotmats_t_1 = _rots_diffuse_mask(rotmats_t_1, rotmats_1, motif_mask)

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj = (trans_t_2, rotmats_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj
        trans_t_1 = _trans_diffuse_mask(trans_t_1, trans_1, motif_mask)
        trans_t_1[:, motif_mask_exp == 0] = trans_m

        rotmats_t_1 = _rots_diffuse_mask(rotmats_t_1, rotmats_1, motif_mask)
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        pred_trans_1 = _trans_diffuse_mask(pred_trans_1, trans_1, motif_mask)
        pred_rotmats_1 = _rots_diffuse_mask(pred_rotmats_1, rotmats_1, motif_mask)
        prot_traj = (pred_trans_1, pred_rotmats_1)

        # Convert trajectories to atom37.
        return (pred_trans_1, pred_rotmats_1)

# DIFFERENCE BETWEEN MISSING RESIDUES AND MOTIF RESIDUES HOW YOU MASK THEM AND ALSO HOW YOU PROCESS FEATURES FOR THEM
    def compute_force(x, eps = 1e-5):
        pass

    def compute_torque(r, eps = 1e-5):
        pass

    def _trans_leapfrog_step(self, d_t, trans_pred, trans_prev, velocity_prev, eps = 1e-5):
        t = 1 - eps
        g_sq = 2*t/(1-t)
        f = -1/(1-t)
        trans_vf = (trans_pred - trans_prev) / (1 - t)
        a = 2/g_sq*(f-trans_vf)*KBT/MASS
        velocity_new = velocity_prev + a*d_t
        trans_new = trans_prev + velocity_new*d_t
        
        return trans_new, velocity_new
    
    def _rots_leapfrog_step(self, d_t, rots_pred, rots_prev, omega_prev, eps = 1e-5):
        scaling = self._rots_cfg.exp_rate
        rots_vf = scaling * so3_utils.calc_rot_vf(rots_prev, rots_pred)
        t = 1 - eps
        g_sq = 2*t/(1-t)
        tau = 2/g_sq*(rots_vf)*KBT
        
        I_in = I_inv.unsqueeze(0).unsqueeze(0).expand(1, 108, 3, 3)
        I_in = I_in.to(rots_pred.device)
        alpha = torch.einsum('...ij,...j->...i', I_in, tau)
        omega_new = omega_prev + alpha * d_t
        rots_new = so3_utils.geodesic_t(
            d_t, rots_pred, rots_prev, omega_new)
        
        return rots_new, omega_new
        
    """

    """
    def sample_traj(self,
            batch,
            model,
            eps = 1e-5
            ):
        num_batch, num_res = batch['res_mask'].shape
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        motif_mask = batch['motif_mask']
        batch['res_mask'] = res_mask
        trans_1 = batch['trans_1']
        rotmats_1 = batch['rotmats_1']

        
        #res_mask = torch.ones(1, batch['num_res'], device=self._device)
        #batch['res_mask'] = res_mask
        #batch['motif_mask'] = res_mask
        #num_batch, num_res = batch['res_mask'].shape
        #motif_mask = batch['motif_mask']
        #batch['res_mask'] = res_mask
        #trans_1 = batch['trans_1']
        #rotmats_1 = batch['rotmats_1']
        
        motif_mask_exp = motif_mask.squeeze(0)
        trans_m = trans_1[:,motif_mask_exp == 0] - trans_1[:,motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_1.device)
        
        init_trans = trans_1
        init_rotmats = rotmats_1
        #init_trans, init_rotmats = self._sample_init(
            #batch, model
        #)

        init_velocity= torch.zeros_like(trans_1, device = trans_1.device)
        init_omega = torch.zeros_like(trans_1, device = trans_1.device)

            
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_traj_timesteps)
        
        t_1 = ts[0]
        prot_traj = [(init_trans, init_rotmats)]
        prot_vel = [(init_velocity, init_omega)]
        clean_traj = []
        for t_2 in range(0):
        #for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            velocity_t_1, omega_t_1 = prot_vel[-1]

            trans_t_1 = _trans_diffuse_mask(trans_t_1, trans_1, motif_mask)
            trans_t_1[:,motif_mask_exp == 0] = trans_m
            rotmats_t_1 = _rots_diffuse_mask(rotmats_t_1, rotmats_1, motif_mask)

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * (1 - eps)
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            #d_t = t_2 - t_1
            d_t = 2e-3 #2fs (units in ps)
            trans_t_2, velocity_t_2 = self._trans_leapfrog_step(d_t, pred_trans_1, trans_t_1, velocity_t_1, eps = 1e-5)
            rotmats_t_2, omega_t_2 = self._rots_leapfrog_step(d_t, pred_rotmats_1, rotmats_t_1, omega_t_1, eps = 1e-5)
            prot_traj.append((trans_t_2, rotmats_t_2))
            prot_vel.append((velocity_t_2, omega_t_2))
            t_1 = t_2


        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj
        
    """
    """
    def _rots_leapfrog_step(self, d_t, rots_pred, rots_prev, omega_prev, I, eps = 1e-5):


    def leapfrog_step(x, v, r, omega, model_out, delta_t, m, I):
    
        # Compute forces and torques based on current positions and rotations
        force = compute_force(x) / m
        torque = compute_torque(r) @ I.inverse()

        # Update velocities and angur velocities
        v_new = v + force * delta_t
        omega_new = omega + torque * delta_t

        # Update positions and rotations
        x_new = x + v_new * delta_t
        r_new = r @ torch.matrix_exp(omega_new * delta_t)

        return x_new, v_new, r_new, omega_new

    def leapfrog_integration(x_t, r_t, d_t, model_out):
        pass
    def compute_loss(traj, x_t, r_t, model_out, dt):
        pass
    """
    def _sample_init(
            self,
            batch,
            model,
        ):
        num_batch, num_res = batch['res_mask'].shape
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        batch['res_mask'] = res_mask

        # Set-up initial prior samples
        trans_0 = _centered_gaussian(
            num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        #rotmats_0 = self.igso3.sample(
            #torch.tensor([1.5]),
            #num_batch*num_res
        #).to(self._device)
        #rotmats_0 = rotmats_0.reshape(num_batch, num_res, 3, 3)

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]
        prot_traj = (trans_0, rotmats_0)
        for t_2 in ts[1:]:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            prot_traj = (trans_t_2, rotmats_t_2)
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj

        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']

        prot_traj = (pred_trans_1, pred_rotmats_1)

        # Convert trajectories to atom37.
        return (pred_trans_1, pred_rotmats_1)
    
    def adjust_adaptive_factor(self):
        if self._adaptive_step_counter % self._num_steps_per_decay == 0:
            self._adaptive_factor *= 0.8 

    def _trans_leapfrog_step(self, trans_pred, trans_prev, velocity_prev):
        #trans_vf = (trans_pred - trans_prev) * one_minus_t_inv
        #f = trans_pred * m
        #a = -2/g_sq*(f-trans_vf)*KBT/MASS
        score = (trans_pred*t - trans_prev)/((1-t)**2)*A
        #score = trans_pred*t - trans_prev
        a = +score*KBT/MASS
        #a = torch.clamp(a, min=-0.08, max=0.08)
        velocity_new = velocity_prev + a*dt
        trans_new = trans_prev + velocity_new*dt
        
        return trans_new, velocity_new, a, trans_pred*t - trans_prev
    
    def _init_velocity_step(self, trans_pred, trans_prev, rots_prev, rots_pred, velocity_prev, omega_prev):
        #trans_vf = (trans_pred - trans_prev) * one_minus_t_inv
        #f = trans_pred * m
        #a = -2/g_sq*(f-trans_vf)*KBT/MASS
        score = (trans_pred*t - trans_prev)/((1-t)**2)*A
        #score = trans_pred*t - trans_prev
        a = +score*KBT/MASS
        #a = torch.clamp(a, min=-0.08, max=0.08)
        velocity_new = velocity_prev - a*dt/2

        #scaling = self._rots_cfg.exp_rate
        scaling = 1/25
        #scaling = 1/247
        rots_vf = scaling * so3_utils.calc_rot_vf(rots_prev, rots_pred)
        tau = +2/g_sq*(rots_vf)*KBT
        
        I_in = I_inv.unsqueeze(0).unsqueeze(0).expand(1, 108, 3, 3)
        ##I_in = I_inv.unsqueeze(0).unsqueeze(0).expand(1, 40, 3, 3)
        I_in = I_in.to(rots_pred.device)
        alpha = torch.einsum('...ij,...j->...i', I_in, tau)
        omega_new = omega_prev - alpha * dt/2

        return velocity_new, omega_new
        
    
    def _rots_leapfrog_step(self, rots_pred, rots_prev, omega_prev):
        scaling = self._rots_cfg.exp_rate
        #scaling = 5
        #scaling = 1
        scaling = 1/25
        #scaling = 1/147
        rots_vf = scaling * so3_utils.calc_rot_vf(rots_prev, rots_pred)
        tau = +2/g_sq*(rots_vf)*KBT
        
        I_in = I_inv.unsqueeze(0).unsqueeze(0).expand(1, 108, 3, 3)
        ##I_in = I_inv.unsqueeze(0).unsqueeze(0).expand(1, 40, 3, 3)
        I_in = I_in.to(rots_pred.device)
        alpha = torch.einsum('...ij,...j->...i', I_in, tau)
        omega_new = omega_prev + alpha * dt
        rots_new = so3_utils.geodesic_t(
            dt, rots_pred, rots_prev, omega_new)
        
        return rots_new, omega_new, alpha
    def sample_traj(self,
            batch,
            init_velocity,
            init_omega,
            model,
            ):
        num_batch, num_res = batch['res_mask'].shape
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        batch['res_mask'] = res_mask
        trans_1 = batch['trans_1']
        rotmats_1 = batch['rotmats_1']
        
        init_trans = trans_1
        init_rotmats = rotmats_1
        init_trans, init_rotmats = self._sample_init(
            batch, model
        )
        init_velocity= torch.zeros_like(trans_1, device = trans_1.device)
        init_omega = torch.zeros_like(trans_1, device = trans_1.device)
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * (1 - eps)

        batch['trans_t'] = trans_1
        batch['rotmats_t'] = rotmats_1
        with torch.no_grad():
            model_out = model(batch)
        # Process model output.
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        init_velocity, init_omega = self._init_velocity_step(pred_trans_1, trans_1, 
                                                 pred_rotmats_1, rotmats_1, 
                                                 init_velocity, init_omega)
        
        prot_traj = [(init_trans, init_rotmats)]
        prot_prev = (init_trans, init_rotmats)
        prot_vel = (init_velocity, init_omega)
        accelerations = []
        alphas = []
        velocities = []
        diff = []
        velocities.append(init_velocity.cpu().numpy())
        omegas = [init_omega.cpu().numpy()]
        
        #clean_traj = []
        step_count = 0
        for _ in range(30000):
        #for _ in ts[1:]:
            step_count += 1
            #self._adaptive_step_counter = step_count
            if step_count % 100 == 0:
                print(f"Step: {step_count}/100_000")

            #if step_count % self._num_steps_per_decay == 0:
                #self.adjust_adaptive_factor()
            # Run model.
            trans_t_1, rotmats_t_1 = prot_prev
            velocity_t_1, omega_t_1 = prot_vel

            #trans_t_1 = _trans_diffuse_mask(trans_t_1, trans_1, motif_mask)
            #trans_t_1[:,motif_mask_exp == 0] = trans_m
            #rotmats_t_1 = _rots_diffuse_mask(rotmats_t_1, rotmats_1, motif_mask)

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            with torch.no_grad():
                model_out = model(batch)
            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            #clean_traj.append(
                #(pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            #)
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            # Take reverse step
            #d_t = t_2 - t_1
            
            trans_t_2, velocity_t_2, a, dif = self._trans_leapfrog_step(pred_trans_1, trans_t_1, velocity_t_1)
            rotmats_t_2, omega_t_2, alpha = self._rots_leapfrog_step(pred_rotmats_1, rotmats_t_1, omega_t_1)
            #rotmats_t_2 = rotmats_t_1
            #omega_t_2 = omega_t_1
            #alpha = torch.zeros_like(velocity_t_2)
            bb_mean = torch.sum(trans_t_2, dim = 1)/108
            ##bb_mean = torch.sum(trans_t_2, dim = 1)/40
            trans_t_2 = trans_t_2 - bb_mean
     
            #trans_t_2[:,motif_mask_exp == 0] = trans_t_2[:,motif_mask_exp == 0] - trans_t_2[:,motif_mask_exp == 0].mean(dim=1, keepdim=True).to(trans_1.device)
            prot_prev = (trans_t_2, rotmats_t_2)
            prot_vel = (velocity_t_2, omega_t_2)
            
            if step_count == 0 or step_count % 10 == 0:
                prot_traj.append((trans_t_2, rotmats_t_2))
            """
                #accelerations.append((a, alpha))
                #velocities.append((velocity_t_2, omega_t_2))
                velocities.append(velocity_t_2.cpu().numpy())  # Convert to numpy array and append
                accelerations.append(a.cpu().numpy())
                omegas.append(omega_t_2.cpu().numpy()) 
                alphas.append(alpha.cpu().numpy()) 
                diff.append(dif.cpu().numpy())
            """
            #prot_traj.append((trans_t_2, rotmats_t_2))
                #accelerations.append((a, alpha))
                #velocities.append((velocity_t_2, omega_t_2))
            #velocities.append(velocity_t_2.cpu().numpy())  # Convert to numpy array and append
            #accelerations.append(a.cpu().numpy())
            #diff.append(dif.cpu().numpy())
            #omegas.append(omega_t_2.cpu().numpy()) 
            #alphas.append(alpha.cpu().numpy())
        
            """
            velocities_array = np.array(velocities)  # Shape will be [len(velocities), 108, 3]
            omegas_array = np.array(omegas)          # Shape will be [len(omegas), 108, 3]
            accelerations_array = np.array(accelerations)
            alphas_array = np.array(alphas)
            diff_array = np.array(diff)

        np.save('protein-frame-flow/data/velocities_new50.npy', velocities_array)  # Save velocities
        np.save('protein-frame-flow/data/omegas_new50.npy', omegas_array) 
        np.save('protein-frame-flow/data/accelerations_new50.npy', accelerations_array) 
        np.save('protein-frame-flow/data/alphas_new50.npy', alphas_array)
        np.save('protein-frame-flow/data/diff_new50.npy', diff_array)
        """
            #prot_traj.append((trans_t_2, rotmats_t_2))
            #prot_vel.append((velocity_t_2, omega_t_2))


        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        #acc37_traj = all_atom.transrot_to_atom37(accelerations, res_mask)
        #vel37_traj = all_atom.transrot_to_atom37(velocities, res_mask)
        #clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        #return atom37_traj, clean_atom37_traj, clean_traj
        #return atom37_traj, vel37_traj, ()
        return atom37_traj, atom37_traj, ()
