import ffmpegio
import os
import numpy as np
import vispy.scene
from vispy import app
from vispy.color import ColorArray
from vispy.visuals import transforms
from vispy.scene.visuals import XYZAxis

# os.makedirs("scripts/viz/imgs/", exist_ok=True)
img_array = []


class Visualizer:
    def __init__(self, pos_file: str, groups_file: str, env: str):
        self.env = env
        self.canvas = vispy.scene.SceneCanvas(
            keys="interactive", show=True, bgcolor="white", title="Something"
        )
        view = self.canvas.central_widget.add_view()
        view.camera = vispy.scene.cameras.TurntableCamera(
            fov=20, azimuth=60, elevation=20, distance=8, up="+y"
        )
        view.camera.set_range(x=(0, 1), y=(0, 1), z=(0, 1))
        self.pos = np.load(pos_file, mmap_mode="r")
        self.groups = np.load(groups_file, mmap_mode="r")
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
        timer = app.Timer()
        timer.connect(self._update)
        self.img_array = []
        timer.start(interval=1.0 / 60.0, iterations=n_frames)
        self.canvas.show()
        app.run()


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
        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
        self.markers.set_data(
            x_norm,
            edge_color="black",
            face_color=color_array,
        )
        img = self.canvas.render()
        self.img_array.append(img)
        print(f"Iteration {event.iteration + 1}/{self.n_frames} Done!")
        if event.iteration == self.n_frames - 1:
            img_array = np.array(self.img_array)
            ffmpegio.video.write(
                self.out_file,
                60,
                img_array,
                show_log=True,
                overwrite=True,
                pix_fmt="yuv420p",
            )
            print("Recorded Dynamics!")
            app.quit()


if __name__ == "__main__":
    visualizer = FluidLabVisualizer(
        pos_file="data/data_Pouring/train/0/x.npy",
        groups_file="data/data_Pouring/train/0/stat.npy",
        env="Pouring",
    )
    visualizer.animate(
        10, "black", "white", out_file="scripts/viz/imgs/pouring_pred.mp4"
    )
