import numpy as np
 #modified by yu
# from dreamerv3.embodied.replay import CuriousReplay


class CountBasedReplay:

    def __init__(self, *args, **kwargs):
        from dreamerv3.embodied.replay import CuriousReplay   #modified by yu
        super().__init__(*args, **kwargs)
        self.should_track_visit_counts = True

    @staticmethod
    def _calculate_priority_score(model_loss, visit_count, hyper):
        return hyper["c"] * np.power(hyper["beta"], visit_count) + hyper["epsilon"]
