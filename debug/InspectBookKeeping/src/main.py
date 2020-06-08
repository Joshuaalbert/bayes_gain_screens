from bayes_gain_screens.datapack import DataPack
from bayes_gain_screens.misc import great_circle_sep
from bayes_gain_screens import TEC_CONV
import numpy as np
import pylab as plt
import os, glob
import argparse


def _setup_dir(d):
    d = os.path.abspath(os.path.expanduser(d))
    os.makedirs(d, exist_ok=True)
    return d


def wrap(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def inspect_tec_inference(dds4_h5parm, dds5_h5parm, plot_dir):
    """
    * Verify that smooth solutions are equal to tec + smoothed[ref].
    * Verify that residuals of raw - smooth is like constant.

    :param dds4_h5parm:
    :param dds5_h5parm:
    :param plot_dir:
    :return:
    """
    dp4 = DataPack(dds4_h5parm, readonly=True)
    dp4.select(pol=0)
    phase, axes = dp4.phase
    Npol, Nd, Na, Nf, Nt = phase.shape
    _, freqs = dp4.get_freqs(axes['freq'])
    tec_conv = TEC_CONV / freqs

    dp5 = DataPack(dds5_h5parm, readonly=True)
    dp5.current_solset = 'directionally_referenced'
    dp5.select(pol=0)
    tec, axes = dp5.tec

    dp5.current_solset = 'smoothed000'
    dp5.select(pol=0)
    phase_smooth, axes = dp5.phase
    phase_mod = tec[..., None, :] * tec_conv[:, None] + phase_smooth[:, 0:1, ...]

    print("Verifying that raw is similar to smoothed")

    def _plot_raw_minus_smoothed000(diff_phase):
        for d in range(Nd)[::5]:
            for a in range(Na)[1::2]:
                plt.imshow(diff_phase[d, a, :, :], aspect='auto', origin='lower')
                plt.colorbar()
                plt.savefig(os.path.join(plot_dir,
                                         "raw_minus_smooth000_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    diff_phase = wrap(wrap(phase) - wrap(phase_smooth))[0, ...]
    print("mean(RAW - smooth000) = {}".format(np.mean(diff_phase)))
    print("max(RAW - smooth000) = {}".format(np.max(np.abs(diff_phase))))
    print("std(RAW - smooth000) = {}".format(np.std(diff_phase)))
    _plot_raw_minus_smoothed000(diff_phase)

    print("Verifying that smooth000 solutions are equal to tec + smoothed[ref].")

    def if_smooth_not_tec_only(diff_phase):
        print('Smoothed000 minus tec is not zero. Max res: {}'.format(np.max(np.abs(diff_phase))))
        for d in range(Nd)[::5]:
            for a in range(Na)[1::2]:
                plt.imshow(diff_phase[d, a, :, :], aspect='auto', origin='lower')
                plt.colorbar()
                plt.savefig(os.path.join(plot_dir,
                                         "smooth000_minus_tec_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    diff_phase = wrap(wrap(phase_smooth) - wrap(phase_mod))[0, ...]
    if not np.all(np.isclose(diff_phase, np.zeros_like(diff_phase), atol=3e-5)):
        if_smooth_not_tec_only(diff_phase)


def inspect_slow_solutions(slow_preapply, dds4_h5parm, dds5_h5parm, dds6_h5parm, dds7_h5parm, dds8_h5parm, plot_dir):
    """
    * Verify pre-apply solutions are correct.
    * Verify slow solutions plus pre-apply (tec + smooth[ref]) resemble raw phase
    * Verify slow solutions resemble const

    :param dds4_h5parm:
    :param dds5_h5parm:
    :param dds7_h5parm:
    :param plot_dir:
    :return:
    """
    print("Verifying pre-apply solutions are correct.")
    # 2,2,Nd,Na,Nf,Nt
    pre_apply_sols = np.angle(np.load(slow_preapply)['Sols']['G'].T)
    assert np.all(pre_apply_sols[1, 0, ...] == 0. + 0.j), "YX should be zero"
    assert np.all(pre_apply_sols[0, 1, ...] == 0. + 0.j), "XY should be zero"
    assert np.all(pre_apply_sols[0, 0, ...] == pre_apply_sols[1, 1, ...]), "preapply pol should be equal"

    dp5 = DataPack(dds5_h5parm, readonly=True)
    dp5.current_solset = 'directionally_referenced'
    dp5.select(pol=0)
    const, _ = dp5.const
    const = wrap(const)
    dp5.current_solset = 'smoothed000'
    dp5.select(pol=0)
    phase_smooth, axes = dp5.phase
    amp_smooth, _ = dp5.amplitude
    Npol, Nd, Na, Nf, Nt = phase_smooth.shape
    diff_smooth_preapply = wrap(wrap(phase_smooth[0, ...]) - wrap(pre_apply_sols[0, 0, ...]))

    assert np.all(np.isclose(diff_smooth_preapply, np.zeros_like(diff_smooth_preapply),
                             atol=1e-5)), "smoothed phase should be equal to preapply 00. max residual {}".format(
        np.abs(diff_smooth_preapply).max())

    print("Verifying slow solutions plus pre-apply resemble raw phase")

    dp4 = DataPack(dds4_h5parm, readonly=True)
    dp4.select(pol=0)
    phase_raw, axes = dp4.phase
    _, raw_directions = dp4.get_directions(axes['dir'])
    raw_directions = np.stack([raw_directions.ra.deg, raw_directions.dec.deg], axis=-1)

    dp7 = DataPack(dds7_h5parm, readonly=True)
    dp7.current_solset = 'sol000'
    dp7.select(pol=0)
    phase_slow, axes_slow = dp7.phase
    amp_slow, axes_slow = dp7.amplitude
    _, slow_directions = dp7.get_directions(axes['dir'])
    slow_directions = np.stack([slow_directions.ra.deg, slow_directions.dec.deg], axis=-1)

    assert np.all(
        np.isclose(slow_directions, raw_directions)), 'directions of smoothed and slow sols should be the same'

    assert np.all(np.isclose(phase_slow[0, ...], phase_slow[-1, ...])), "slow solution polarisations should be the same"

    time_map = np.array([np.argmin(np.abs(axes_slow['time'] - t)) for t in axes['time']])
    print("Time Map: ", time_map)
    phase_slow = phase_slow[..., time_map]
    amp_slow = amp_slow[..., time_map]

    slow_plus_smoothed = phase_slow + phase_smooth

    diff_raw_minus_slow_plus_smoothed = wrap(wrap(phase_raw[0, ...]) - wrap(slow_plus_smoothed[0, ...]))

    print("mean(RAW - (slow + smooothed000)) {}".format(np.mean(diff_raw_minus_slow_plus_smoothed)))
    print("max(RAW - (slow + smooothed000)) {}".format(np.max(np.abs(diff_raw_minus_slow_plus_smoothed))))
    print("std(RAW - (slow + smooothed000)) {}".format(np.std(diff_raw_minus_slow_plus_smoothed)))

    def _plot_raw_minus_slow_plus_smoothed(diff_raw_minus_slow_plus_smoothed):
        for d in range(Nd)[::5]:
            for a in range(Na)[1::2]:
                plt.imshow(diff_raw_minus_slow_plus_smoothed[d, a, :, :], aspect='auto', origin='lower')
                plt.colorbar()
                plt.savefig(os.path.join(plot_dir,
                                         "raw_minus_slow_plus_smoothed000_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    _plot_raw_minus_slow_plus_smoothed(diff_raw_minus_slow_plus_smoothed)

    print("Verifying slow solutions resemble const")

    diff_slow_const = wrap(wrap(phase_slow[0, ...]) - wrap(const[0, ..., None, :]))
    print("mean(slow - const) {}".format(np.mean(diff_slow_const)))
    print("max(slow - const) {}".format(np.max(np.abs(diff_slow_const))))
    print("std(slow - const) {}".format(np.std(diff_slow_const)))

    def _plot_diff_slow_const(diff_slow_const):
        for d in range(Nd)[::5]:
            for a in range(Na)[1::2]:
                plt.imshow(diff_slow_const[d, a, :, :], aspect='auto', origin='lower')
                plt.colorbar()
                plt.savefig(os.path.join(plot_dir,
                                         "slow_minus_const_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    _plot_diff_slow_const(diff_slow_const)

    print("Verifying screen_slow000 equals slow plus smoothed000")

    dp8 = DataPack(dds8_h5parm, readonly=True)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0, dir=slice(0, 45, 1))
    phase_screen, screen_axes = dp8.phase
    _, screen_directions = dp8.get_directions(screen_axes['dir'])
    screen_directions = np.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=-1)

    assert np.all(
        np.isclose(screen_directions, raw_directions)), 'directions of screen and slow sols should be the same'

    diff_screen_cals_minus_slow_plus_smoothed = wrap(wrap(phase_screen[0, ...]) - wrap(slow_plus_smoothed[0, ...]))

    def _plot_screen_minus_slow_plus_smoothed(diff_screen_cals_minus_slow_plus_smoothed):
        print("screen is not equal to slow plus smoothed000. Max res {}".format(
            np.max(np.abs(diff_screen_cals_minus_slow_plus_smoothed))))
        for d in range(Nd)[::5]:
            for a in range(Na)[1::2]:
                ax = plt.subplot(1, 3, 1)
                img = ax.imshow(diff_screen_cals_minus_slow_plus_smoothed[d, a, :, :], aspect='auto', origin='lower')
                ax.set_title("screen cals minus smooth000 + slow")
                plt.colorbar(img)
                ax = plt.subplot(1, 3, 2)
                ax.plot(const[0, d, a, :])
                ax.set_title('const')
                ax = plt.subplot(1, 3, 3)
                img = ax.imshow(phase_slow[0, d, a, :, :], aspect='auto', origin='lower')
                ax.set_title("slow")
                plt.colorbar(img)
                plt.savefig(os.path.join(plot_dir,
                                         "screen_minus_slow_plus_smoothed000_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    if not np.all(np.isclose(diff_screen_cals_minus_slow_plus_smoothed,
                             np.zeros_like(diff_screen_cals_minus_slow_plus_smoothed), atol=1e-5)):
        _plot_screen_minus_slow_plus_smoothed(diff_screen_cals_minus_slow_plus_smoothed)

    print("Verifying smoothed_slow000 equals slow plus smoothed000")

    dp8 = DataPack(dds8_h5parm, readonly=True)
    dp8.current_solset = 'smoothed_slow000'
    dp8.select(pol=0)
    phase_smoothed_slow, smoothed_slow_directions_axes = dp8.phase
    _, smoothed_slow_directions = dp8.get_directions(smoothed_slow_directions_axes['dir'])
    smoothed_slow_directions = np.stack([smoothed_slow_directions.ra.deg, smoothed_slow_directions.dec.deg], axis=-1)

    assert np.all(np.isclose(smoothed_slow_directions,
                             raw_directions)), 'directions of smoothe_slow_axes and slow sols should be the same'

    diff_smoothed_slow_cals_minus_slow_plus_smoothed = wrap(
        wrap(phase_smoothed_slow[0, ...]) - wrap(slow_plus_smoothed[0, ...]))

    def _plot_smoothed_slow_minus_slow_plus_smoothed(diff_smoothed_slow_cals_minus_slow_plus_smoothed):
        print("smoothed_slow000 is not equal to slow plus smoothed000. Max res {}".format(
            np.max(np.abs(diff_smoothed_slow_cals_minus_slow_plus_smoothed))))
        for d in range(Nd)[::5]:
            for a in range(Na)[1::2]:
                ax = plt.subplot(1, 3, 1)
                img = ax.imshow(diff_smoothed_slow_cals_minus_slow_plus_smoothed[d, a, :, :], aspect='auto',
                                origin='lower')
                ax.set_title("smoothed_slow000 cals minus smooth000 + slow")
                plt.colorbar(img)
                ax = plt.subplot(1, 3, 2)
                ax.plot(const[0, d, a, :])
                ax.set_title('const')
                ax = plt.subplot(1, 3, 3)
                img = ax.imshow(phase_slow[0, d, a, :, :], aspect='auto', origin='lower')
                ax.set_title("slow")
                plt.colorbar(img)
                plt.savefig(os.path.join(plot_dir,
                                         "smoothed_slow_minus_slow_plus_smoothed000_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    if not np.all(np.isclose(diff_smoothed_slow_cals_minus_slow_plus_smoothed,
                             np.zeros_like(diff_smoothed_slow_cals_minus_slow_plus_smoothed), atol=1e-5)):
        _plot_smoothed_slow_minus_slow_plus_smoothed(diff_smoothed_slow_cals_minus_slow_plus_smoothed)

    print("Verifying that screen_slow000 is screen+slow for non-calibrators")

    dp6 = DataPack(dds6_h5parm, readonly=True)
    dp6.current_solset = 'screen_posterior'
    dp6.select(pol=0, dir=slice(45, None, 1))
    phase_screen, screen_axes = dp6.phase
    _, screen_directions = dp6.get_directions(screen_axes['dir'])
    screen_directions = np.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=-1)

    dir_map = np.array([np.argmin(great_circle_sep(slow_directions[:, 0], slow_directions[:, 1], ra, dec))
                        for (ra, dec) in zip(screen_directions[:, 0], screen_directions[:, 1])])

    phase_screen_plus_slow = phase_screen + phase_slow[:, dir_map, ...]

    dp8 = DataPack(dds8_h5parm, readonly=True)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0, dir=slice(45, None, 1))
    phase_screen_slow, _ = dp8.phase

    diff_non_cal_screen_slow000_minus_screen_plus_slow = wrap(
        wrap(phase_screen_slow[0, ...]) - wrap(phase_screen_plus_slow[0, ...]))

    def _plot_non_cal_screen_slow000_minus_screen_plus_slow(diff_non_cal_screen_slow000_minus_screen_plus_slow):
        print("Non-call screen_slow000 is not equal to slow plus screen_posterior. Max res {}".format(
            np.max(np.abs(diff_non_cal_screen_slow000_minus_screen_plus_slow))))
        Nd = diff_non_cal_screen_slow000_minus_screen_plus_slow.shape[1]
        for d in range(Nd)[::20]:
            for a in range(Na)[1::2]:
                ax = plt.subplot(1, 2, 1)
                img = ax.imshow(diff_non_cal_screen_slow000_minus_screen_plus_slow[d, a, :, :], aspect='auto',
                                origin='lower')
                ax.set_title("screen_slow000 non-cal minus screen + slow")
                plt.colorbar(img)
                ax = plt.subplot(1, 2, 2)
                img = ax.imshow(phase_screen[0, d, a, :, :], aspect='auto',
                                origin='lower')
                ax.set_title("screen_posterior non_cal")
                plt.colorbar(img)
                plt.savefig(os.path.join(plot_dir,
                                         "non_cal_screen_slow000_minus_screen_plus_slow_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    if not np.all(np.isclose(diff_non_cal_screen_slow000_minus_screen_plus_slow,
                             np.zeros_like(diff_non_cal_screen_slow000_minus_screen_plus_slow), atol=1e-5)):
        _plot_non_cal_screen_slow000_minus_screen_plus_slow(diff_non_cal_screen_slow000_minus_screen_plus_slow)

    print("Setting correct screen_slow000")

    dp6 = DataPack(dds6_h5parm, readonly=True)
    dp6.current_solset = 'screen_posterior'
    dp6.select(pol=0)
    phase_screen, screen_axes = dp6.phase
    _, screen_directions = dp6.get_directions(screen_axes['dir'])
    screen_directions = np.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=-1)

    dir_map = np.array([np.argmin(great_circle_sep(slow_directions[:, 0], slow_directions[:, 1], ra, dec))
                        for (ra, dec) in zip(screen_directions[:, 0], screen_directions[:, 1])])

    phase_screen_plus_slow = phase_screen + phase_slow[:, dir_map, ...]
    Ncal = slow_directions.shape[0]
    phase_screen_plus_slow[:, :Ncal, ...] = slow_plus_smoothed
    amp_screen_plus_slow = amp_smooth[:, dir_map, ...] * amp_slow[:, dir_map, ...]

    dp8 = DataPack(dds8_h5parm, readonly=False)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0)

    dp8.phase = phase_screen_plus_slow
    dp8.amplitude = amp_screen_plus_slow

    dp8.current_solset = 'smoothed_slow000'
    dp8.select(pol=0)

    dp8.phase = slow_plus_smoothed
    dp8.amplitude = amp_smooth * amp_slow

    print("Verifying that screen_slow000 is screen+slow for all. It must be, we just set it!")

    dp6 = DataPack(dds6_h5parm, readonly=True)
    dp6.current_solset = 'screen_posterior'
    dp6.select(pol=0)
    phase_screen, screen_axes = dp6.phase
    _, screen_directions = dp6.get_directions(screen_axes['dir'])
    screen_directions = np.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=-1)

    dir_map = np.array([np.argmin(great_circle_sep(slow_directions[:, 0], slow_directions[:, 1], ra, dec))
                        for (ra, dec) in zip(screen_directions[:, 0], screen_directions[:, 1])])

    phase_screen_plus_slow = phase_screen + phase_slow[:, dir_map, ...]

    dp8 = DataPack(dds8_h5parm, readonly=True)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0)
    phase_screen_slow, _ = dp8.phase

    diff_non_cal_screen_slow000_minus_screen_plus_slow = wrap(
        wrap(phase_screen_slow[0, ...]) - wrap(phase_screen_plus_slow[0, ...]))

    def _plot_non_cal_screen_slow000_minus_screen_plus_slow(diff_non_cal_screen_slow000_minus_screen_plus_slow):
        print("Non-call screen_slow000 is not equal to slow plus screen_posterior. Max res {}".format(
            np.max(np.abs(diff_non_cal_screen_slow000_minus_screen_plus_slow))))
        Nd = diff_non_cal_screen_slow000_minus_screen_plus_slow.shape[1]
        for d in range(Nd)[::25]:
            for a in range(Na)[1::2]:
                ax = plt.subplot(1, 2, 1)
                img = ax.imshow(diff_non_cal_screen_slow000_minus_screen_plus_slow[d, a, :, :], aspect='auto',
                                origin='lower')
                ax.set_title("screen_slow000 non-cal minus screen + slow")
                plt.colorbar(img)
                ax = plt.subplot(1, 2, 2)
                img = ax.imshow(phase_screen[0, d, a, :, :], aspect='auto',
                                origin='lower')
                ax.set_title("screen_posterior non_cal")
                plt.colorbar(img)
                plt.savefig(os.path.join(plot_dir,
                                         "postset_check_non_cal_screen_slow000_minus_screen_plus_slow_dir{:02d}_ant{:02d}.png".format(
                                             d, a)))
                plt.close('all')

    if not np.all(np.isclose(diff_non_cal_screen_slow000_minus_screen_plus_slow,
                             np.zeros_like(diff_non_cal_screen_slow000_minus_screen_plus_slow), atol=1e-5)):
        _plot_non_cal_screen_slow000_minus_screen_plus_slow(diff_non_cal_screen_slow000_minus_screen_plus_slow)


def inspect_bug(dds5_h5parm, dds6_h5parm, dds7_h5parm, dds8_h5parm, plot_dir):
    """
    * Verify pre-apply solutions are correct.
    * Verify slow solutions plus pre-apply (tec + smooth[ref]) resemble raw phase
    * Verify slow solutions resemble const

    :param dds4_h5parm:
    :param dds5_h5parm:
    :param dds7_h5parm:
    :param plot_dir:
    :return:
    """
    print("Using:")
    for f in [dds5_h5parm, dds6_h5parm, dds7_h5parm, dds8_h5parm]:
        print(f)

    dp5 = DataPack(dds5_h5parm, readonly=True)
    dp5.current_solset = 'smoothed000'
    dp5.select(pol=0)
    phase_smooth, axes = dp5.phase
    amp_smooth, _ = dp5.amplitude
    Npol, Nd, Na, Nf, Nt = phase_smooth.shape

    dp7 = DataPack(dds7_h5parm, readonly=True)
    dp7.current_solset = 'sol000'
    dp7.select(pol=0)
    phase_slow, axes_slow = dp7.phase
    amp_slow, axes_slow = dp7.amplitude
    _, slow_directions = dp7.get_directions(axes['dir'])
    slow_directions = np.stack([slow_directions.ra.deg, slow_directions.dec.deg], axis=-1)

    time_map = np.array([np.argmin(np.abs(axes_slow['time'] - t)) for t in axes['time']])
    print("Time Map: ", time_map)

    phase_slow = phase_slow[..., time_map]
    amp_slow = amp_slow[..., time_map]

    assert phase_slow.shape == phase_smooth.shape
    assert amp_slow.shape == amp_smooth.shape

    phase_slow_plus_smoothed = phase_slow + phase_smooth
    amp_slow_times_smoothed = amp_slow * amp_smooth
    assert np.all(np.isfinite(amp_slow_times_smoothed))
    print('amp_slow_times_smoothed', amp_slow_times_smoothed.min(), amp_slow_times_smoothed.max())
    print('amp_slow', amp_slow.min(), amp_slow.max())
    print('amp_smooth', amp_smooth.min(), amp_smooth.max())

    dp6 = DataPack(dds6_h5parm, readonly=True)
    dp6.current_solset = 'screen_posterior'
    dp6.select(pol=0)
    phase_screen, screen_axes = dp6.phase
    _, screen_directions = dp6.get_directions(screen_axes['dir'])
    screen_directions = np.stack([screen_directions.ra.deg, screen_directions.dec.deg], axis=-1)

    dir_map = np.array([np.argmin(great_circle_sep(slow_directions[:, 0], slow_directions[:, 1], ra, dec))
                        for (ra, dec) in zip(screen_directions[:, 0], screen_directions[:, 1])])

    print("Direction map: ", dir_map)

    phase_screen_plus_slow = phase_screen + phase_slow[:, dir_map, ...]

    print("Verifying that screen_slow000 is correct.")

    def _verify_same(diff, what, should_be_what):
        res = np.max(np.abs(diff))
        if res > 1e-5:
            print(f"{what} is not equal to {should_be_what}. Max res {res}")
            Nd = diff.shape[1]
            for d in range(Nd)[::25]:
                for a in range(Na)[1::2]:
                    img = plt.imshow(diff[d, a, :, :], aspect='auto',
                                     origin='lower')
                    plt.title(f"{what} minus {should_be_what}")
                    plt.colorbar(img)
                    plt.savefig(os.path.join(plot_dir,
                                             "{}_minus_{}_dir{:02d}_ant{:02d}.png".format(
                                                 what, should_be_what,
                                                 d, a)))
                    plt.close('all')

    print("First non-calibrators...")

    dp8 = DataPack(dds8_h5parm, readonly=True)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0)
    phase_screen_slow000, _ = dp8.phase
    amp_screen_slow000, _ = dp8.amplitude

    Ncal = phase_slow.shape[1]
    diff = wrap(wrap(phase_screen_slow000[0, Ncal:, ...]) - wrap(phase_screen_plus_slow[0, Ncal:, ...]))

    _verify_same(diff, 'non_cal_screen_slow000_phase', 'screen_plus_slow')

    diff = amp_screen_slow000[0, ...] - amp_slow_times_smoothed[0, dir_map, ...]

    _verify_same(diff, 'screen_slow000_amp', 'slow_times_smoothed_amp')

    print("Now calibrators")

    diff = wrap(wrap(phase_screen_slow000[0, :Ncal, ...]) - wrap(phase_slow_plus_smoothed[0, ...]))

    _verify_same(diff, 'cal_screen_slow000_phase', 'smoothed000_plus_slow')

    print("Setting correct screen_slow000")

    dp8 = DataPack(dds8_h5parm, readonly=False)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0, dir=slice(Ncal, None, 1))
    dp8.phase = phase_screen_plus_slow[0:1, Ncal:, ...]
    dp8.select(pol=0, dir=slice(None, Ncal, 1))
    dp8.phase = phase_slow_plus_smoothed[0:1, ...]
    dp8.select(pol=0, dir=None)
    dp8.amplitude = amp_slow_times_smoothed[0:1, dir_map, ...]

    print("Verifying that screen_slow000 is correct. Must be we just set it!")

    dp8 = DataPack(dds8_h5parm, readonly=True)
    dp8.current_solset = 'screen_slow000'
    dp8.select(pol=0)
    phase_screen_slow000, _ = dp8.phase
    amp_screen_slow000, _ = dp8.amplitude

    Ncal = phase_slow.shape[1]
    diff = wrap(wrap(phase_screen_slow000[0, Ncal:, ...]) - wrap(phase_screen_plus_slow[0, Ncal:, ...]))

    _verify_same(diff, 'non_cal_screen_slow000_phase', 'screen_plus_slow')

    diff = amp_screen_slow000[0, ...] - amp_slow_times_smoothed[0, dir_map, ...]

    _verify_same(diff, 'screen_slow000_amp', 'slow_times_smoothed_amp')

    print("Now calibrators")

    diff = wrap(wrap(phase_screen_slow000[0, :Ncal, ...]) - wrap(phase_slow_plus_smoothed[0, ...]))

    _verify_same(diff, 'cal_screen_slow000_phase', 'smoothed000_plus_slow')


def main(root_dir, plot_dir):
    plot_dir = _setup_dir(plot_dir)
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    solve_dds4_dir = os.path.join(root_dir, 'solve_dds4')
    dds4_h5parm = glob.glob(os.path.join(solve_dds4_dir, '*DDS4*h5'))[0]
    tec_and_smooth_dir = os.path.join(root_dir, 'tec_inference_and_smooth')
    dds5_h5parm = glob.glob(os.path.join(tec_and_smooth_dir, '*DDS5*h5'))[0]
    infer_screen_dir = os.path.join(root_dir, 'infer_screen')
    dds6_h5parm = glob.glob(os.path.join(infer_screen_dir, '*DDS6*h5'))[0]
    slow_dir = os.path.join(root_dir, 'slow_solve_dds4')
    dds7_h5parm = glob.glob(os.path.join(slow_dir, '*DDS7*h5'))[0]
    slow_preapply = glob.glob(os.path.join(slow_dir, "*DDS5*sols.npz"))[0]
    merge_dir = os.path.join(root_dir, 'merge_slow')
    dds8_h5parm = glob.glob(os.path.join(merge_dir, '*DDS8*h5'))[0]

    # inspect_tec_inference(dds4_h5parm, dds5_h5parm, _setup_dir(os.path.join(plot_dir, 'tec_inspect')))
    # inspect_slow_solutions(slow_preapply, dds4_h5parm, dds5_h5parm, dds6_h5parm, dds7_h5parm, dds8_h5parm,
    #                        _setup_dir(os.path.join(plot_dir, 'slow_inspect')))
    inspect_bug(dds5_h5parm, dds6_h5parm, dds7_h5parm, dds8_h5parm, _setup_dir(os.path.join(plot_dir, 'bug_inspect')))


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--root_dir', help='Root dir of pipeline run.', type=str, required=True)
    parser.add_argument('--plot_dir', help='Plot dir of this.', type=str, required=False, default='./plots_manual')


if __name__ == '__main__':
    import sys

    sys.argv.append('--root_dir')
    sys.argv.append('/home/albert/store/root_solve_bias_90_allslow/L342938')
    parser = argparse.ArgumentParser(
        description='Tec outlier detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("\t{} -> {}".format(option, value))
    main(**vars(flags))
