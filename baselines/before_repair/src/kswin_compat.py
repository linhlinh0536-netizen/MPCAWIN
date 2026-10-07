import numpy as np
from scipy import stats


class KSWIN:

    def __init__(
        self,
        alpha=0.005,
        window_size=100,
        stat_size=30,
        data=None
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size

        self.change_detected = False
        self.p_value = None
        self.statistic = None

        # Lưu 2 sample gần nhất để debug
        self.last_reference = None
        self.last_recent = None

        self.n = 0

        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        if window_size < stat_size:
            raise ValueError(
                "stat_size must be <= window_size"
            )

        if isinstance(data, np.ndarray):
            self.window = data
        else:
            self.window = np.array([])


    def add_element(self, input_value):

        self.n += 1
        current_length = len(self.window)

        # reset debug value của lần test hiện tại
        self.p_value = None
        self.statistic = None
        self.last_reference = None
        self.last_recent = None

        if current_length >= self.window_size:

            # bỏ phần tử cũ nhất
            self.window = np.delete(
                self.window,
                0
            )

            # phần recent
            recent_window = self.window[
                -self.stat_size:
            ]

            # lấy mẫu từ phần reference cũ
            reference_pool = self.window[
                :-self.stat_size
            ]

            rnd_window = np.random.choice(
                reference_pool,
                self.stat_size
            )

            # lưu để debug
            self.last_reference = rnd_window.copy()
            self.last_recent = recent_window.copy()

            self.statistic, self.p_value = stats.ks_2samp(
                rnd_window,
                recent_window,
                mode="exact"
            )

            if (
                self.p_value <= self.alpha
                and self.statistic > 0.1
            ):
                self.change_detected = True

                self.window = self.window[
                    -self.stat_size:
                ]

            else:
                self.change_detected = False

        else:
            self.change_detected = False

        # KSWIN gốc luôn append phần tử mới
        self.window = np.concatenate(
            [self.window, [input_value]]
        )


    def detected_change(self):
        return self.change_detected


    def reset(self):
        self.window = np.array([])
        self.change_detected = False

        self.p_value = None
        self.statistic = None

        self.last_reference = None
        self.last_recent = None

        self.n = 0