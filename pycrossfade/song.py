import numpy as np
from pathlib import Path
from . import utils


_beat_tracker = None


class Song():
    def __init__(self, filepath=None, cache_dir=None):
        self.filepath = filepath
        self.cache_dir = cache_dir
        self.audio = None
        self.sample_rate = 44100
        self.beats = None
        self.downbeats = None

        if filepath is not None:
            self.song_name, self.song_format = self.get_song_name_and_format()
            self.load_song_audio()
            self.load_beats()

    #def plot_downbeats(self, start_dbeat, end_dbeat, plot_name='', color='red'):
    #    import matplotlib.pyplot as plt
    #    plt.rcParams['figure.figsize'] = (20, 9) 
    #    dbeats = self.get_downbeats()
    #    start_idx, end_idx = dbeats[start_dbeat], dbeats[end_dbeat]
    #    selected_dbeats = dbeats[start_dbeat:end_dbeat+1] - start_idx
    #    plt.plot(self.audio[start_idx: end_idx])
    #    for dbeat in selected_dbeats:
    #        plt.axvline(dbeat, color=color)
    #    plt.title(plot_name)
    #    plotname = ''.join(plot_name.split(' '))
    #    plt.savefig(f'{plotname}.png')
        

    def load_song_audio(self):
        self.audio = utils.load_audio(self.filepath)

    def get_song_name_and_format(self):
        """Return the filename stem and extension without breaking dotted titles."""
        path = Path(self.filepath)
        return path.stem, path.suffix.lstrip('.')

    def annotate_beats(self):
        tracker = get_beat_tracker()
        beats, downbeats = tracker(self.filepath)
        return utils.make_beat_annotations(beats, downbeats)

    def get_downbeats(self):
        if self.downbeats is not None:
            return self.downbeats

        beats = self.beats
        dbeats = []
        for beat_sec, beat_num in beats:
            if beat_num == 1:
                dbeats.append(beat_sec)
        dbeats_time_to_audio_index = np.array(dbeats, dtype=float) * self.sample_rate
        self.downbeats = np.array(dbeats_time_to_audio_index, dtype=int)
        return self.downbeats

    def load_beats(self):
        cache_key = utils.content_hash(self.filepath)
        cached_beats = utils.load_cached_beats(cache_key, self.cache_dir)
        if cached_beats is not None:
            self.beats = cached_beats
            return

        self.beats = self.annotate_beats()
        utils.save_cached_beats(cache_key, self.beats, self.filepath, self.cache_dir)


def get_beat_tracker():
    """Create the Beat This! model once per process.

    CPU inference is the portable default. Applications that need GPU inference
    can replace this factory with their own configured tracker.
    """
    global _beat_tracker
    if _beat_tracker is None:
        from beat_this.inference import File2Beats
        _beat_tracker = File2Beats(checkpoint_path='final0', device='cpu', dbn=False)
    return _beat_tracker
