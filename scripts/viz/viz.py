import torch
import ffmpegio
import argparse
import os
import time
import einops
from PIL import Image
import numpy as np
import vispy.scene
import vispy.app
from vispy import app
from vispy.color import ColorArray
from vispy.visuals import transforms
from vispy.scene.visuals import XYZAxis


def rotation_matrix_from_quaternion(params):
    # params: (B * n_instance) x 4
    # w, x, y, z

    one = torch.ones(1, 1)
    zero = torch.zeros(1, 1)
    use_gpu = True
    if use_gpu:
        one = one.cuda()
        zero = zero.cuda()

    # multiply the rotation matrix from the right-hand side
    # the matrix should be the transpose of the conventional one

    # Reference
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

    params = params / torch.norm(params, dim=1, keepdim=True)
    w, x, y, z = (
        params[:, 0].view(-1, 1, 1),
        params[:, 1].view(-1, 1, 1),
        params[:, 2].view(-1, 1, 1),
        params[:, 3].view(-1, 1, 1),
    )

    rot = torch.cat(
        (
            torch.cat(
                (
                    one - y * y * 2 - z * z * 2,
                    x * y * 2 + z * w * 2,
                    x * z * 2 - y * w * 2,
                ),
                2,
            ),
            torch.cat(
                (
                    x * y * 2 - z * w * 2,
                    one - x * x * 2 - z * z * 2,
                    y * z * 2 + x * w * 2,
                ),
                2,
            ),
            torch.cat(
                (
                    x * z * 2 + y * w * 2,
                    y * z * 2 - x * w * 2,
                    one - x * x * 2 - y * y * 2,
                ),
                2,
            ),
        ),
        1,
    )

    # rot: (B * n_instance) x 3 x 3
    return rot


class Visualizer:
    def __init__(self, pos_file: str, groups_file: str, env: str):
        self.app = app.use_app()
        self.env = env
        self.canvas = vispy.scene.SceneCanvas(
            keys="interactive",
            show=True,
            bgcolor="white",
            title="Something",
            app=self.app,
        )
        view = self.canvas.central_widget.add_view()
        view.camera = vispy.scene.cameras.TurntableCamera(
            fov=30, azimuth=60, elevation=20, distance=8, up="+y"
        )
        view.camera.set_range(x=(0, 1), y=(0, 1), z=(0, 1))
        self.pos_file = pos_file
        self.pos = np.load(pos_file, mmap_mode="r")
        mins = einops.reduce(self.pos, "t n s -> s", "min")
        maxs = einops.reduce(self.pos, "t n s -> s", "max")
        self.pos = (self.pos - mins) / (maxs - mins)

        self.groups = np.load(groups_file, mmap_mode="r")[0]
        self.markers = vispy.scene.Markers()
        self.max_frames = self.pos.shape[0]
        self.markers.antialias = 0
        view.add(self.markers)

    def _update(self, event):
        raise NotImplementedError

    def animate(self, n_frames: int, color_zero: str, color_one: str, out_file: str):
        assert (
            n_frames <= self.pos.shape[0]
        ), f"{self.env} supports a maximum of {self.max_frames} frames!"

        assert (
            not hasattr(self, "img_array") or len(self.img_array) == 0
        ), "Visualizer is already animating something..."

        self.n_frames = n_frames
        self.color_zero = color_zero
        self.color_one = color_one
        self.out_file = out_file
        print(f"Attempting to animate positions found in {self.pos_file}")
        self.timer = app.Timer(app=self.app)
        self.timer.connect(self._update)
        self.img_array = []
        self.timer.start(interval=1.0 / 60.0, iterations=n_frames)
        self.canvas.show()


class FluidLabVisualizer(Visualizer):
    def __init__(self, pos_file: str, groups_file: str, env: str):
        super().__init__(pos_file, groups_file, env)

    def _update(self, event):
        pos = self.pos[event.iteration]
        indices_non_neg = np.where(
            np.logical_and(np.any(pos >= -1, axis=1), np.any(pos <= 1, axis=1))
            # squeeze out the actual indices & exclude the agent pos
        )[0][:-1]
        groups_t = self.groups[indices_non_neg]
        colors = np.zeros_like(groups_t, dtype=object)
        colors[groups_t == 0] = self.color_zero
        colors[groups_t == 1] = self.color_one
        color_array = ColorArray(list(colors))
        x = pos[indices_non_neg]
        self.markers.set_data(
            x,
            edge_color="black",
            face_color=color_array,
        )
        img = self.canvas.render()
        self.img_array.append(img)
        # print(event.iteration)
        if event.iteration == self.n_frames - 1:
            img_array = np.array(self.img_array)
            ffmpegio.video.write(
                self.out_file,
                90,
                img_array,
                show_log=True,
                overwrite=True,
                pix_fmt="yuv420p",
            )
            print("Recorded Dynamics!")


class QuatVisualizer(Visualizer):
    def __init__(self, pos_file: str, groups_file: str, env: str):
        self.boundary = torch.Tensor([[0.05, 0.05, 0.05], [0.95, 0.95, 0.95]]).cuda()
        super().__init__(pos_file, groups_file, env)

    def _update(self, event):
        quat = np.zeros((1, 4))
        loaded_quat = self.pos[event.iteration][-1]
        quat[0, [0, 2, 3]] = loaded_quat
        quat = torch.Tensor(quat).cuda()
        rot_matrix = rotation_matrix_from_quaternion(quat)
        res = (self.boundary @ rot_matrix).squeeze(0)
        # self.boundary = res  
        # self.markers.set_data(self.boundary.cpu().numpy())
        self.markers.set_data(res.cpu().numpy())
        img = self.canvas.render()
        self.img_array.append(img)
        if event.iteration == self.n_frames - 1:
            img_array = np.array(self.img_array)
            ffmpegio.video.write(
                self.out_file,
                90,
                img_array,
                show_log=True,
                overwrite=True,
                pix_fmt="yuv420p",
            )
            print("Recorded Dynamics!")


def gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inf", type=str)
    parser.add_argument("--outf_gt", type=str)
    parser.add_argument("--outf_pred", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = gen_args()
    for i in range(1):
        visualizer_gt = FluidLabVisualizer(
            pos_file=f"{args.inf}/0_gt.npy",
            groups_file=f"{args.inf}/0_groups.npy",
            env="Pouring",
        )
        visualizer_gt.animate(
            1000, "black", "white", out_file=f"scripts/viz/imgs/{args.outf_gt}.mp4"
        )
        # app.run()
        visualizer_pred = FluidLabVisualizer(
            pos_file=f"{args.inf}/0_pred.npy",
            groups_file=f"{args.inf}/0_groups.npy",
            env="Pouring",
        )
        visualizer_pred.animate(
            1000, "black", "white", out_file=f"scripts/viz/imgs/{args.outf_pred}.mp4"
        )
    # visualizer_gt = QuatVisualizer(
    #     pos_file=f"{args.inf}/0_gt.npy",
    #     groups_file=f"{args.inf}/0_groups.npy",
    #     env="Pouring",
    # )
    # visualizer_gt.animate(
    #     1000, "black", "white", out_file=f"scripts/viz/imgs/{args.outf_pred}.mp4"
    # )
    app.run()
