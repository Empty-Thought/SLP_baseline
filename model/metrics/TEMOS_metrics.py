import torch
from torchmetrics import Metric

def l2_norm(x1, x2, dim):
    return torch.linalg.norm(x1 - x2, dim=dim, ord=2)

def variance(x, T, dim):
    mean = x.mean(dim=dim)
    out = ((x - mean.unsqueeze(dim))**2).sum(dim=dim) / (T - 1)
    return out

class ComputeMetrices(Metric):

    def __init__(self, jointstype=None, force_in_meter=None, dist_sync_on_step=None, **kwargs):
        super().__init__(**kwargs)

        self.rifke = None

        self.force_in_meter = force_in_meter

        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_sequence", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # APE
        self.add_state("APE_pose", default=torch.zeros(51), dist_reduce_fx="sum")
        self.add_state("APE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.APE_metrics = ["APE_pose", "APE_joints"]

        # AVE
        self.add_state("AVE_pose", default=torch.zeros(51), dist_reduce_fx="sum")
        self.add_state("AVE_joints", default=torch.zeros(21), dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_pose", "AVE_joints"]

        self.metrics = self.APE_metrics + self.AVE_metrics
    
    def compute(self):
        count = self.count
        count_seq = self.count_sequence

        APE_metrics = {metric: getattr (self, metric) / count for metric in self.APE_metrics}

        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count
        # Remove arrays
        APE_metrics.pop("APE_pose")
        APE_metrics.pop("APE_joints")


        AVE_metrics = {metric: getattr (self, metric) / count_seq for metric in self.AVE_metrics}

        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq

        # Remove arrays
        AVE_metrics.pop("AVE_pose")
        AVE_metrics.pop("AVE_joints")
        return {**APE_metrics, **AVE_metrics}


    def update(self, pose_pred, pose_gt, lengths):
        

        self.count += sum(lengths)
        self.count_sequence += len(lengths)

        pose_pred = self.transform(pose_pred)
        pose_gt = self.transform(pose_gt)
        
        for i in range(len(lengths)):

            self.APE_pose += l2_norm(pose_pred[i], pose_gt[i], dim=2).sum(0)

            pose_sigma_pred = variance(pose_pred[i], lengths[i], dim=0)
            pose_sigma_gt = variance(pose_gt[i], lengths[i], dim=0)

            self.AVE_pose += l2_norm(pose_sigma_pred, pose_sigma_gt, dim=1)

    def transform(self, joints):
        
        B, T, F = joints.shape
        J_num = F // 3

        joints = joints.reshape(B, T, J_num, 3)

        return joints



        pass



if __name__ == "__main__":
    # test l2_norm


    # test variance
    x = torch.randn(3, 4, 5)
    T = x.shape[1]
    print(variance(x, T, dim=1).shape)  # [3, 5]