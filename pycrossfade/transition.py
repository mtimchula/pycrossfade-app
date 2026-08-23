import numpy as np
from .utils import *
from .song import Song


def _audio_rms(audio):
    values = np.asarray(audio, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        return 0.0
    return float(np.sqrt(np.mean(np.square(values)) + 1e-12))


def _pad_audio(audio, length):
    values = np.asarray(audio)
    missing = length - len(values)
    if missing <= 0:
        return values[:length]
    shape = (missing,) + values.shape[1:]
    return np.concatenate((values, np.zeros(shape, dtype=values.dtype)), axis=0)


def validate_song_for_transition(song, minimum_downbeats, role):
    audio = np.asarray(song.audio)
    downbeats = np.asarray(song.get_downbeats(), dtype=int)
    if audio.ndim not in (1, 2) or not len(audio):
        raise ValueError(f'{role} track has no supported audio samples.')
    if not np.isfinite(audio).all():
        raise ValueError(f'{role} track contains non-finite audio samples.')
    if len(downbeats) < minimum_downbeats:
        raise ValueError(
            f'{role} track needs at least {minimum_downbeats} downbeats; '
            f'analysis found {len(downbeats)}.'
        )
    if np.any(np.diff(downbeats) <= 0):
        raise ValueError(f'{role} track downbeats must be strictly increasing.')
    if downbeats[0] < 0 or downbeats[-1] >= len(audio):
        raise ValueError(f'{role} track downbeats fall outside its audio bounds.')
    return downbeats

def crop_audio_and_dbeats(song, start_dbeat, end_dbeat):
    audio = song.audio
    song_dbeats = song.get_downbeats()
    len_dbeats = len(song_dbeats)

    # Supporting negative indexing
    if start_dbeat < 0:
        start_dbeat = len_dbeats + start_dbeat
    if end_dbeat < 0:
        end_dbeat = len_dbeats + end_dbeat

    if start_dbeat >= len_dbeats or end_dbeat >= len_dbeats:  # or start_dbeat >= end_dbeat:
        raise Exception(f"Given start_dbeat({start_dbeat}) and/or end_dbeat({end_dbeat}) are not compatible.")
    
    start_dbeat_value = song_dbeats[start_dbeat]
    audio_start_idx, audio_end_idx = song_dbeats[start_dbeat], song_dbeats[end_dbeat]
    cropped_audio = audio[audio_start_idx: audio_end_idx]
    cropped_dbeats = song_dbeats[start_dbeat:end_dbeat] - start_dbeat_value

    new_song = Song()
    new_song.audio = cropped_audio
    new_song.downbeats = cropped_dbeats
    return new_song


def time_stretch_gradually_in_downbeats(song, final_factor):
    # Since we are time stretching *in-between* down beats, dbeat array has to
    # include the +1 dbeat in itself
    audio = song.audio
    dbeats = song.get_downbeats()

    if not np.isfinite(final_factor) or final_factor <= 0:
        raise ValueError(f'Invalid time-stretch factor: {final_factor}.')
    if np.isclose(final_factor, 1.0):
        return audio

    boundaries = np.unique(np.concatenate((np.asarray(dbeats, dtype=int), [len(audio)])))
    if len(boundaries) < 2:
        raise ValueError('Gradual time stretching requires at least one audio interval.')
    ts_factors = np.linspace(1.0, final_factor, len(boundaries) - 1, dtype=float)

    time_stretched_audio_slices = []
    for i, factor in enumerate(ts_factors):
        audio_slice = audio[boundaries[i]:boundaries[i + 1]]
        if len(audio_slice):
            time_stretched_audio_slices.append(time_stretch(audio_slice, factor))

    if not time_stretched_audio_slices:
        raise ValueError('Gradual time stretching produced no audio intervals.')
    output = np.concatenate(time_stretched_audio_slices, axis=0)
    return output


def find_compatible_slave_start(master_dbeats, slave_dbeats, len_crossfade,
                                max_search_seconds=64,
                                max_tempo_ratio=1.25):
    """Choose a stable entry point when a song's opening is tempo-incompatible.

    Beat trackers can be uncertain during intros and may mark subdivisions as
    downbeats. Starting a crossfade there can force an extreme time stretch.
    This function retains the opening by default, but searches a bounded part
    of the incoming track for a consecutive window with compatible bar timing
    when the opening would exceed ``max_tempo_ratio``.
    """
    window_size = len_crossfade
    if len(master_dbeats) < window_size + 1 or len(slave_dbeats) < window_size + 1:
        return 0

    master_intervals = np.diff(master_dbeats[-(window_size + 1):]).astype(float)
    opening_intervals = np.diff(slave_dbeats[:window_size + 1]).astype(float)
    opening_ratios = opening_intervals / master_intervals
    if safe_tempo_window(
        opening_ratios,
        minimum=1 / max_tempo_ratio,
        maximum=max_tempo_ratio,
    ):
        return 0

    max_index = min(
        len(slave_dbeats) - window_size - 1,
        int(np.searchsorted(slave_dbeats, slave_dbeats[0] + max_search_seconds * 44100)),
    )
    best_index = 0
    best_score = float('inf')
    for index in range(max_index + 1):
        slave_intervals = np.diff(slave_dbeats[index:index + window_size + 1]).astype(float)
        ratios = slave_intervals / master_intervals
        if np.any(ratios < 1 / max_tempo_ratio) or np.any(ratios > max_tempo_ratio):
            continue
        score = np.mean(np.abs(np.log(ratios))) + np.std(slave_intervals) / np.mean(slave_intervals)
        if score < best_score:
            best_score = score
            best_index = index
    return best_index

def beatmatch_to_slave(master_song, slave_song):
    master_audio = master_song.audio
    master_dbeats = master_song.get_downbeats()
    slave_audio = slave_song.audio
    slave_dbeats= slave_song.get_downbeats()

    if len(master_dbeats) != len(slave_dbeats):
        raise Exception(f"master_dbeats({len(master_dbeats)}) and slave_dbeats({len(slave_dbeats)}) is not same length")

    len_beatmatch_dbeats = len(master_dbeats)


    
    # Time stretching between every dbeat, according their respective time difference
    time_stretched_master_fadeout_audio_fragments = []
    master_next_idx = None
    for i in range(len_beatmatch_dbeats - 1):
        # these are dbeats
        master_cur_idx, master_next_idx = master_dbeats[i], master_dbeats[i + 1]
        slave_cur_idx, slave_next_idx = slave_dbeats[i], slave_dbeats[i + 1]

        # getting the next_dbeat_index - current_dbeat_index difference
        master_dbeat_diff_idx = master_next_idx - master_cur_idx
        slave_dbeat_diff_idx = slave_next_idx - slave_cur_idx

        # calculating the time stretch factor
        ts_factor = master_dbeat_diff_idx / slave_dbeat_diff_idx

        # getting the masters audio fragments for that downbeat indices
        master_audio_frag = master_audio[master_cur_idx:master_next_idx]

        if not np.isfinite(ts_factor) or ts_factor <= 0:
            raise ValueError(f'Invalid per-bar time-stretch factor: {ts_factor}.')
        ts_maf = time_stretch(master_audio_frag, ts_factor)

        # when time stretching with floating point factors, created audio can be more or less in length
        # so we are fixing that here. Its usually 20-50 frame indices difference, so its inaudible to cut it off
        if len(ts_maf) > slave_dbeat_diff_idx:
            ts_maf = ts_maf[:slave_dbeat_diff_idx]
        elif len(ts_maf) < slave_dbeat_diff_idx:
            ts_maf = _pad_audio(ts_maf, slave_dbeat_diff_idx)

        # adding the current dbeats time stretched master audio fragment to the list
        # we will add them together later.
        time_stretched_master_fadeout_audio_fragments.append(ts_maf)

    # ------ Adding the last part ------
    master_cur_idx, master_next_idx = master_dbeats[-1], len(master_audio)
    slave_cur_idx, slave_next_idx = slave_dbeats[-1], len(slave_audio)
    # getting the next_dbeat_index - current_dbeat_index difference
    master_dbeat_diff_idx = master_next_idx - master_cur_idx
    slave_dbeat_diff_idx = slave_next_idx - slave_cur_idx
    # calculating the time stretch factor
    if slave_dbeat_diff_idx <= 0 or master_dbeat_diff_idx <= 0:
        raise ValueError('Beat-match tail interval must contain audio in both tracks.')
    ts_factor = master_dbeat_diff_idx / slave_dbeat_diff_idx

    # getting the masters audio fragments for that downbeat indices
    master_audio_frag = master_audio[master_cur_idx:master_next_idx]

    if not np.isfinite(ts_factor) or ts_factor <= 0:
        raise ValueError(f'Invalid tail time-stretch factor: {ts_factor}.')
    ts_maf = time_stretch(master_audio_frag, ts_factor)


    # when time stretching with floating point factors, created audio can be more or less in length
    # so we are fixing that here. Its usually 20-50 frame indices difference, so its inaudible to cut it off
    if len(ts_maf) > slave_dbeat_diff_idx:
        ts_maf = ts_maf[:slave_dbeat_diff_idx]
    elif len(ts_maf) < slave_dbeat_diff_idx:
        ts_maf = _pad_audio(ts_maf, slave_dbeat_diff_idx)

    # adding the current dbeats time stretched master audio fragment to the list
    # we will add them together later.
    time_stretched_master_fadeout_audio_fragments.append(ts_maf)

    # ------ END Adding the last part ------

    # putting time_stretched_master_fadeout_audio_fragments together
    master_beatmatched_to_slave_audio = np.concatenate(
        time_stretched_master_fadeout_audio_fragments, axis=0
    )
    # must be same length: master_beatmatched_to_slave_audio, slave_audio
    return master_beatmatched_to_slave_audio, slave_audio




def crossfade(master_song, slave_song, len_crossfade, len_time_stretch,
              return_audio=True, slave_start_dbeat=None,
              overlay_rough_transition=False):
    if len_crossfade < 1 or len_time_stretch < 0:
        raise ValueError('Crossfade must be positive and time-stretch length cannot be negative.')
    # We are getting the required song partitions and their respective dbeats from SongPartition class
    master_p_audio = master_song.audio
    master_p_dbeats = validate_song_for_transition(
        master_song, len_crossfade + len_time_stretch + 1, 'Outgoing'
    )
    slave_p_audio = slave_song.audio
    slave_p_dbeats = validate_song_for_transition(
        slave_song, len_crossfade + 1, 'Incoming'
    )

    if slave_start_dbeat is None:
        slave_start_dbeat = find_compatible_slave_start(
            master_p_dbeats, slave_p_dbeats, len_crossfade
        )

    master_window = np.diff(master_p_dbeats[-(len_crossfade + 1):])
    slave_window = np.diff(
        slave_p_dbeats[slave_start_dbeat:slave_start_dbeat + len_crossfade + 1]
    )
    bar_ratios = master_window.astype(float) / slave_window.astype(float)
    tempo_ratio = float(np.median(bar_ratios))
    if not safe_tempo_window(bar_ratios):
        return fallback_crossfade(
            master_song,
            slave_song,
            len_crossfade,
            slave_start_dbeat,
            return_audio,
            overlay_rough_transition,
            tempo_ratio,
            'no_candidate_with_all_bar_ratios_between_0.80_and_1.25',
        )

    master_phrase = master_p_audio[
        master_p_dbeats[-(len_crossfade + 1)]:master_p_dbeats[-1]
    ]
    slave_phrase = slave_p_audio[
        slave_p_dbeats[slave_start_dbeat]:
        slave_p_dbeats[slave_start_dbeat + len_crossfade]
    ]
    master_rms = _audio_rms(master_phrase)
    slave_rms = _audio_rms(slave_phrase)
    transition_gain = 1.0
    if master_rms > 1e-6 and slave_rms > 1e-6:
        transition_gain = float(np.clip(
            master_rms / slave_rms,
            10 ** (-3 / 20),
            10 ** (3 / 20),
        ))
        slave_song.audio = np.asarray(slave_song.audio, dtype=np.float32) * transition_gain
        slave_p_audio = slave_song.audio

    # calculate the factor of time stretching according to first
    # downbeat difference of master and slaves in crossfade
    crossfade_master_first_dbeat_diff = master_p_dbeats[(-1 * len_crossfade) + 1] - master_p_dbeats[-1 * len_crossfade]
    crossfade_slave_first_dbeat_diff = (
        slave_p_dbeats[slave_start_dbeat + 1] - slave_p_dbeats[slave_start_dbeat]
    )
    ts_final_factor = crossfade_master_first_dbeat_diff / crossfade_slave_first_dbeat_diff

    # -- TIME STRETCHING --

    ts_dbeat_start = -1 * (len_crossfade + len_time_stretch)
    ts_dbeat_end = (-1 * len_crossfade) + 1
    ts_song = Song()
    ts_song.audio, ts_song.downbeats = master_p_audio, master_p_dbeats
    ts_cropped_song = crop_audio_and_dbeats(ts_song, ts_dbeat_start, ts_dbeat_end)
    
    time_stretch_audio = time_stretch_gradually_in_downbeats(ts_cropped_song, ts_final_factor)
    ts_start_idx = master_p_dbeats[ts_dbeat_start]

    # -- END TIME STRETCHING --

    # -- CROSSFADING --

    master_dbeats_start = len(master_p_dbeats) - len_crossfade - 1
    master_dbeats_end = len(master_p_dbeats) - 1
    master_fadeout_song = Song()
    master_fadeout_song.audio, master_fadeout_song.downbeats = master_p_audio, master_p_dbeats
    master_fadeout_cropped_song = crop_audio_and_dbeats(master_fadeout_song,
                                                                          master_dbeats_start,
                                                                          master_dbeats_end)
    slave_dbeats_start = slave_start_dbeat
    slave_dbeats_end = slave_start_dbeat + len_crossfade
    slave_fadein_song = Song()
    slave_fadein_song.audio, slave_fadein_song.downbeats = slave_p_audio, slave_p_dbeats
    slave_fadein_cropped_song = crop_audio_and_dbeats(slave_fadein_song,
                                                    slave_dbeats_start,
                                                    slave_dbeats_end)


    master_fadeout_audio, slave_fadein_audio = beatmatch_to_slave(master_fadeout_cropped_song, slave_fadein_cropped_song)

    assert len(master_fadeout_audio) == len(slave_fadein_audio)

    # Sound Effects for Master
    new_master_fadedout = linear_fade_volume(master_fadeout_audio, start_volume=0.9, end_volume=0.0)
    new_master_fadedout = linear_fade_filter(new_master_fadedout, 'low_shelf', start_volume=0.9, end_volume=0.0)
    new_master_fadedout = linear_fade_filter(new_master_fadedout, 'high_shelf', start_volume=0.9, end_volume=0.0)

    # Sound Effects for Slave
    new_slave_fadedin = linear_fade_volume(slave_fadein_audio, start_volume=0.1, end_volume=1.0)
    new_slave_fadedin = linear_fade_filter(new_slave_fadedin, 'low_shelf', start_volume=0.0, end_volume=1.0)
    new_slave_fadedin = linear_fade_filter(new_slave_fadedin, 'high_shelf', start_volume=0.0, end_volume=1.0)

    crossfade_audio = new_slave_fadedin + new_master_fadedout

    slave_fadein_end_idx = slave_p_dbeats[slave_start_dbeat] + len(new_slave_fadedin)


    if return_audio:
        return np.concatenate([
            master_song.audio[:ts_start_idx],
            time_stretch_audio,
            crossfade_audio,
            slave_song.audio[slave_fadein_end_idx:]
        ])
    else:
        return {'time_stretch_audio': time_stretch_audio,
                'crossfade_audio': crossfade_audio,
                'len_crossfade': len_crossfade,
                'len_time_stretch': len_time_stretch,
                'slave_start_dbeat': slave_start_dbeat,
                'ts_start_idx': ts_start_idx,
                'slave_fadein_end_idx': slave_fadein_end_idx,
                'rough_transition': False,
                'fallback_reason': None,
                'tempo_ratio': tempo_ratio,
                'bar_ratios': [round(float(value), 5) for value in bar_ratios],
                'incoming_gain_db': round(float(20 * np.log10(transition_gain)), 3)}


def crossfade_multiple(song_list, len_crossfade, len_time_stretch,
                       overlay_rough_transitions=False,
                       return_manifest=False):
    if len(song_list) < 2:
        raise ValueError('A mix requires at least two songs.')
    master_song = song_list[0]
    slave_song = song_list[1]
    output_list = []
    transition_manifest = []
    output_length = 0

    # crossfade and crossfade-before
    cf_before = None
    cf = None

    # crossfade between partitions
    for i in range(len(song_list) - 1):
        master_song = song_list[i]
        slave_song = song_list[i + 1]
        cf_before = cf
        cf = crossfade(
            master_song, slave_song, len_crossfade, len_time_stretch,
            return_audio=False,
            overlay_rough_transition=overlay_rough_transitions,
        )

        if i == 0:
            # if its the first song on the list
            master_p_audio_start_idx = 0
            master_p_audio_end_idx = cf['ts_start_idx']

        else:
            # if its not
            master_p_audio_start_idx = cf_before['slave_fadein_end_idx']
            master_p_audio_end_idx = cf['ts_start_idx']

        master_audio = master_song.audio
        prefix = master_audio[master_p_audio_start_idx:master_p_audio_end_idx]
        output_list.append(prefix)
        output_length += len(prefix)
        output_list.append(cf['time_stretch_audio'])
        output_length += len(cf['time_stretch_audio'])
        transition_start = output_length
        output_list.append(cf['crossfade_audio'])
        output_length += len(cf['crossfade_audio'])
        transition_manifest.append({
            'position': i,
            'output_start_sample': transition_start,
            'output_end_sample': output_length,
            'output_start_seconds': round(transition_start / master_song.sample_rate, 3),
            'output_end_seconds': round(output_length / master_song.sample_rate, 3),
            'outgoing_start_sample': int(cf['ts_start_idx']),
            'incoming_start_downbeat': int(cf['slave_start_dbeat']),
            'incoming_end_sample': int(cf['slave_fadein_end_idx']),
            'rough_transition': bool(cf.get('rough_transition', False)),
            'fallback_reason': cf.get('fallback_reason'),
            'tempo_ratio': round(float(cf.get('tempo_ratio', 1.0)), 5),
            'bar_ratios': cf.get('bar_ratios', []),
            'incoming_gain_db': float(cf.get('incoming_gain_db', 0.0)),
        })

    # adding the last part
    slave_audio = slave_song.audio
    output_list.append(slave_audio[cf['slave_fadein_end_idx']:])
    output = np.concatenate(output_list, axis=0)
    if not np.isfinite(output).all():
        raise ValueError('Mix output contains non-finite audio samples.')
    if return_manifest:
        return output, transition_manifest
    return output


def fallback_crossfade(master_song, slave_song, len_crossfade,
                       slave_start_dbeat, return_audio,
                       overlay_horn, tempo_ratio,
                       fallback_reason='unsafe_tempo_window'):
    """Crossfade without stretching when beat annotations imply an unsafe ramp."""
    master_dbeats = master_song.get_downbeats()
    slave_dbeats = slave_song.get_downbeats()
    master_intervals = np.diff(master_dbeats).astype(float)
    stable_intervals = master_intervals[
        (master_intervals >= .5 * master_song.sample_rate)
        & (master_intervals <= 3 * master_song.sample_rate)
    ]
    if not stable_intervals.size:
        stable_intervals = master_intervals
    fade_length = int(np.median(stable_intervals) * len_crossfade)
    slave_start = int(slave_dbeats[slave_start_dbeat])
    fade_length = min(fade_length, len(master_song.audio), len(slave_song.audio) - slave_start)
    if fade_length <= 0:
        raise ValueError('Guarded transition has no usable crossfade audio.')
    master_start = find_outro_crossfade_start(
        master_song.audio,
        master_dbeats,
        fade_length,
        master_song.sample_rate,
    )
    master_end = master_start + fade_length

    phase = np.linspace(0, np.pi / 2, fade_length, dtype=np.float32)
    curve_shape = (fade_length,) + (1,) * (np.asarray(master_song.audio).ndim - 1)
    master_curve = np.cos(phase).reshape(curve_shape)
    slave_curve = np.sin(phase).reshape(curve_shape)
    master_segment = master_song.audio[master_start:master_end]
    slave_segment = slave_song.audio[slave_start:slave_start + fade_length]
    master_rms = _audio_rms(master_segment)
    slave_rms = _audio_rms(slave_segment)
    transition_gain = 1.0
    if master_rms > 1e-6 and slave_rms > 1e-6:
        transition_gain = float(np.clip(
            master_rms / slave_rms,
            10 ** (-3 / 20),
            10 ** (3 / 20),
        ))
        slave_song.audio = np.asarray(slave_song.audio, dtype=np.float32) * transition_gain
        slave_segment = slave_song.audio[slave_start:slave_start + fade_length]
    master_fade = master_segment * master_curve
    slave_fade = slave_segment * slave_curve
    transition_audio = master_fade + slave_fade
    if overlay_horn:
        horn = air_horn(min(fade_length, int(1.6 * master_song.sample_rate)), master_song.sample_rate)
        horn_start = min(int(.35 * master_song.sample_rate), max(0, fade_length - len(horn)))
        if transition_audio.ndim == 2:
            horn = horn[:, np.newaxis]
        transition_audio[horn_start:horn_start + len(horn)] += horn

    details = {
        'time_stretch_audio': np.empty(
            (0,) + transition_audio.shape[1:],
            dtype=transition_audio.dtype,
        ),
        'crossfade_audio': transition_audio,
        'len_crossfade': len_crossfade,
        'len_time_stretch': 0,
        'slave_start_dbeat': slave_start_dbeat,
        'ts_start_idx': master_start,
        'slave_fadein_end_idx': slave_start + fade_length,
        'rough_transition': True,
        'fallback_reason': fallback_reason,
        'tempo_ratio': tempo_ratio,
        'bar_ratios': [],
        'incoming_gain_db': round(float(20 * np.log10(transition_gain)), 3),
    }
    if return_audio:
        return np.concatenate([
            master_song.audio[:master_start], transition_audio,
            slave_song.audio[slave_start + fade_length:],
        ])
    return details


def air_horn(length, sample_rate=44100):
    """Generate a short, license-free DJ air-horn style accent."""
    time = np.arange(length, dtype=np.float32) / sample_rate
    attack = np.minimum(1.0, time / .025)
    release = np.exp(-2.4 * time)
    vibrato = 1.0 + .012 * np.sin(2 * np.pi * 7 * time)
    horn = sum(
        weight * np.sin(2 * np.pi * frequency * vibrato * time)
        for frequency, weight in ((220, .55), (277.18, .30), (329.63, .18))
    )
    return (.20 * attack * release * np.tanh(1.8 * horn)).astype(np.float32)


def safe_tempo_window(bar_ratios, minimum=0.80, maximum=1.25):
    """Reject a transition if even one bar would create an audible speed jump."""
    ratios = np.asarray(bar_ratios, dtype=float)
    return bool(
        ratios.size
        and np.all(np.isfinite(ratios))
        and np.all((ratios >= minimum) & (ratios <= maximum))
    )


def find_outro_crossfade_start(audio, downbeats, fade_length, sample_rate=44100,
                               max_search_seconds=64, minimum_energy_ratio=.30):
    """Choose the latest phrase that stays audible, avoiding silent file tails."""
    audio = np.asarray(audio)
    default_start = max(0, len(audio) - fade_length)
    if not audio.size or fade_length <= 0:
        return default_start
    overall_rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if overall_rms <= 1e-6:
        return default_start

    earliest = max(0, len(audio) - int(max_search_seconds * sample_rate))
    candidates = []
    for downbeat in np.asarray(downbeats, dtype=int):
        end = downbeat + fade_length
        if downbeat < earliest or end > len(audio):
            continue
        quarters = np.array_split(audio[downbeat:end], 4)
        sustained_rms = min(
            float(np.sqrt(np.mean(np.square(quarter)) + 1e-12))
            for quarter in quarters if quarter.size
        )
        if sustained_rms >= overall_rms * minimum_energy_ratio:
            candidates.append(int(downbeat))
    return candidates[-1] if candidates else default_start
