import numpy as np

# from dreamerv3.embodied.replay import CuriousReplay   #modified by yu

 #modified by yu
# class AdversarialReplay(CuriousReplay):
class AdversarialReplay:
    def __init__(self, *args, **kwargs):
        from dreamerv3.embodied.replay import CuriousReplay    #modified by yu
        super().__init__(*args, **kwargs)
        self.should_track_visit_counts = False

    @staticmethod
    def _calculate_priority_score(model_loss, visit_count, hyper):
        return np.power((model_loss + hyper["epsilon"]), hyper["alpha"])
