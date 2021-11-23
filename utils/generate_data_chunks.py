import datetime as dt
import logging

import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot
from tqdm import tqdm


# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

def check_sampling_freq(mag_df, min_sep=None, verbose=True):
    """Determine the sampling frequency from the data

    Compute a weighted-average of the sampling frequency
    present in the time-series data. This is done by taking
    the rolling difference between consecutive datetime indices
    and then binning them up using a method of pd.Series objects.
    Also computes some statistics describing the distribution of
    sampling frequencies.

    Parameters
    ----------
    mag_df : pd.DataFrame
        Pandas dataframe containing the magnetometer data

    min_sep : float
        Minimum separation between two consecutive observations
        to be consider usable for discontinuity identification

    verbose : boolean
        Specifies information on diverging sampling frequencies

    Returns
    -------
    avg_sampling_freq : float
        Weighted average of the sampling frequencies in the dataset

    stats : dict
        Some descriptive statistics for the interval

    """
    # Boolean flag for quality of data in interval
    # Assume its not bad and set to True if it is
    bad = False

    # Compute the time difference between consecutive measurements
    # a_i - a_{i-1} and save the data as dt.timedelta objects
    # rounded to the nearest milisecond
    diff_dt = mag_df.index.to_series().diff(1).round('ms')
    sampling_freqs = diff_dt.value_counts()
    sampling_freqs /= sampling_freqs.sum()

    avg_sampling_freq = 0
    for t, percentage in sampling_freqs.items():
        avg_sampling_freq += t.total_seconds() * percentage


    # Compute the difference in units of seconds so we can compute the RMS
    diff_s = np.array(
        list(
            map(lambda val: val.total_seconds(), diff_dt)
        )
    )

    # Compute the RMS of the observation times to look for gaps in
    # in the observation period
    # t_rms = np.sqrt(
    #     np.nanmean(
    #         np.square(diff_s)
    #     )
    # )
    # # flag that the gaps larger the min_sep.
    # if min_sep is None:
    #     min_sep = 5 * t_rms

    gap_indices = np.where(diff_s > min_sep)[0]
    n_gaps = len(gap_indices)
    # LOG.info(f'{avg_sampling_freq}, {n_gaps}')

    try:
        previous_indices = gap_indices - 1
    except TypeError as e:
        #         LOG.warning(e)
        print(e)
        total_missing = 0

    else:
        gap_durations = mag_df.index[gap_indices] \
                             - mag_df.index[previous_indices]
        total_missing = sum(gap_durations.total_seconds())

    # Compute the duration of the entire interval and determine the coverage
    total_duration = (mag_df.index[-1] - mag_df.index[0]).total_seconds()
    if total_duration == 0:
        print(mag_df)
    coverage = 1 - total_missing / total_duration

    if verbose and coverage < 0.75:
        msg = (
            f"\n Observational coverage: {coverage:0.2%}\n"
            f"Number of data gaps: {n_gaps:0.0f}\n"
            f"Average sampling rate: {avg_sampling_freq:0.5f}"
        )
        LOG.warning(msg)
        bad = True

    stats_data = dict()
    stats_data['average_freq'] = avg_sampling_freq
    stats_data['max_freq'] = sampling_freqs.index.max().total_seconds()
    stats_data['min_freq'] = sampling_freqs.index.min().total_seconds()
    stats_data['n_gaps'] = len(gap_indices)
    stats_data['starttime_gaps'] = [mag_df.index[previous_indices]]
    stats_data['endtime_gaps'] = [mag_df.index[gap_indices]]
    stats_data['gap_durations'] = [gap_durations]
    stats_data['total_missing'] = total_missing
    stats_data['coverage'] = coverage
    return avg_sampling_freq, stats_data, bad

def do_img_transformation(
        df,
        cols=(
                'BTOTAL',
                'BX(RTN)',
                'BY(RTN)',
                'BZ(RTN)',
                'VP_RTN',
                'TEMPERATURE',
                'BETA'
        ),
        threshold='distance',
        percentage=10
):
    rp = RecurrencePlot(
        threshold=threshold,
        percentage=percentage
    )
    img_data = {}
    for col in cols:
        X = df[col].values.reshape(1, -1)
        img = rp.fit_transform(X)
        # invert the image so the edges correspond to distinct features
        # in the timeseries
        inverted_img = np.where(img > 0, 0, 1)
        img_data[col] = inverted_img
    imgs_stacked = np.concatenate(list(img_data.values()))
    return img_data, imgs_stacked


def visualize_chunk_img(imgs, chunk_num=0, icme=0):
    values = imgs.values()
    keys = imgs.keys()
    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(nrows=2, ncols=4, hspace=0.25, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes += [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax, val, key in zip(axes, values, keys):
        ax.imshow(val[0], cmap='binary', origin='lower', aspect='equal')
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )
        ax.set_title(key)
        ax.grid(False)
    fig.suptitle(
        t=f'Chunk Number {chunk_num:0.0f}, ICME={icme:0.0f}',
        x=0.5,
        y=0.95
    )
    return fig


def visualize_chunk_ts(
        raw_df,
        preprocessed_df=None,
        cols=(
                'BTOTAL',
                'BX(RTN)',
                'BY(RTN)',
                'BZ(RTN)',
                'VP_RTN',
                'TEMPERATURE',
                'BETA'
        ),
        chunk_num=0,
        icme=0,
        icme_date=None
):
    fig, axes = plt.subplots(
        nrows=len(cols), ncols=1, figsize=(5,8), sharex=True
    )
    ylims = [
        (-2, 30),
        (-15, 15),
        (-15, 15),
        (-15, 15),
        (200, 800),
        (5e3, 1e6),
        (-5, 75)
    ]
    for ax in axes[:4]:
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    axes[4].yaxis.set_major_locator(plt.MultipleLocator(200))
    axes[-1].yaxis.set_major_locator(plt.MultipleLocator(10))
    for i, (ax, col) in enumerate(zip(axes, cols)):
        ax.tick_params(axis='both', which='major', labelsize=9)
        x = raw_df.index
        y = raw_df[col]
        ax.plot(x, y, lw=0.8, c='r')
        if preprocessed_df is not None:
            x_p = preprocessed_df.index
            y_p = preprocessed_df[col]
            ax.plot(x_p, y_p, lw=1, c='k')
        ax.set_ylabel(col, fontsize=10)
        ax.set_ylim(ylims[i])
        if 'temp' in col.lower():
            ax.set_yscale('log')

        if icme_date is not None:
            ax.axvline(icme_date, ls='-', c='k', lw=1)

    axes[0].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            locator=mdates.HourLocator(),
            formats=['', '', '%m-%d', '%H:%M', '%H:%M', '%S.%f'],
            offset_formats=[
                '%Y',
                '%b',
                '%b %d, %Y',
                '%b %d, %Y',
                '%b %d, %Y',
                '%b %d, %Y'
            ],
            zero_formats=['', '%Y', '%b', '%b-%d', '%H:%M',
                          '%H:%M'],
            show_offset=True
        )
    )
    fig.suptitle(
        f'Chunk Number {chunk_num:0.0f}, ICME={icme:0.0f}',
        x=0.5,
        y=0.95
    )
    return fig


# noinspection PyTupleAssignmentBalance
def create_data_chunks(
        data_df,
        icme_df,
        instrument,
        window_size=dt.timedelta(days=2)):
    """

    Parameters
    ----------
    data_df
    window_size

    Returns
    -------

    """
    start_date = data_df.index[0]
    end_date = data_df.index[-1]

    # current_time = start_date + window_size
    start_interval = start_date - window_size
    end_interval = start_date + window_size
    intervals = []
    # 1 if it contains ICME, 0 otherwise
    stride = dt.timedelta(days=1)
    while start_interval < end_date - window_size:
        intervals.append(slice(start_interval, end_interval))
        start_interval += window_size / 2
        end_interval += window_size / 2
    missing_periods = open(f'../data/st{instrument}_missing_data.txt', 'w')
    intervals_ts_pdf = PdfPages(f'st{instrument}_ts_intervals_visualized.pdf')
    # intervals_img_pdf = PdfPages(f'st{instrument}_img_intervals_visualized.pdf')
    interval_labels = {
        'fname':[],
        'label':[]
    }
    chunk_num = 1
    for i, interval in tqdm(enumerate(intervals), total=len(intervals)):
        st = interval.start.strftime('%Y-%m-%d_%H:%M:%S')
        et = interval.stop.strftime('%Y-%m-%d_%H:%M:%S')
        data_interval_df = data_df[interval]
        if data_interval_df.empty:
            LOG.info('No data... skipping...')
            missing_periods.write(f"{st} {et}\n")
            continue
        icme_interval_df = icme_df[interval]

        # Check to see if there is even data in the interval
        data_span = (
                data_interval_df.index[-1] - data_interval_df.index[0]
        ).total_seconds()
        interval_span = (interval.stop - interval.start).total_seconds()
        overlap = data_span / interval_span
        if data_interval_df.shape[0] <= 100 or overlap < 0.75:
            LOG.info(f'Insufficient data for {st} {et}... skipping...')
            missing_periods.write(f"{st} {et}\n")
            continue

        # Proceed with checking the interval for large data gaps
        fout = (
            f"st{instrument}_ts_interval"
            f"_{st.replace(':','_')}_to_{et.replace(':','_')}.txt"
        )
        label = 0
        # Compute the sampling frequency and the coverage
        avg_sampling_freq, stats_data, bad = check_sampling_freq(
            data_interval_df,
            min_sep=120
        )
        if bad:
            LOG.info(f'Too many data gaps in {st} {et}... skipping...')
            missing_periods.write(f"{st} {et}\n")
            continue

        icme_date = None
        if icme_interval_df.shape[0] >= 1:
            icme_date = icme_interval_df.index[0]
            if icme_date < interval.stop - window_size/3:
                label = 1
                LOG.info(f'Interval {st} {et} contains ICME!')

        # check to see if there is anything missing data and if there is
        # perform linear interpolation to fill the gaps.

        # if stats_data['coverage'] < 1:
        #     LOG.info(
        #         f"Coverage of {stats_data['coverage']:0.2%}... interpolating"
        #     )
        #     data_interval_df = data_interval_df.interpolate(
        #         method='linear', axis=0, inplace=False
        #     )

        smoothed = data_interval_df.rolling(
            '20min', center=True, min_periods=15
        ).mean()
        resampled = smoothed.resample('5min', ).mean()
        imgs_dict, imgs_stacked = do_img_transformation(
            df=resampled.dropna(),
            cols=resampled.columns
        )
        interval_labels['fname'].append(fout)
        interval_labels['label'].append(label)
        resampled.to_csv(
            f"../data/st{instrument}_chunks/{fout}",
            header=True,
            index=True
        )
        # Save the image cube
        fout_img = fout.replace('ts_interval','img_interval')
        fout_img = fout_img.replace('.txt','.npy')
        np.save(
            f"../data/st{instrument}_chunks/{fout_img}",
            imgs_stacked
        )
        # Plot the time series data for each chunk
        # fig = visualize_chunk_ts(
        #     data_interval_df,
        #     resampled,
        #     cols=resampled.columns,
        #     chunk_num=i,
        #     icme=label,
        #     icme_date=icme_date
        # )
        # intervals_ts_pdf.savefig(fig, bbox_inches='tight')

        # Plot the corresponding image representation
        # fig_img = visualize_chunk_img(
        #     imgs_dict,
        #     chunk_num=chunk_num,
        #     icme=label
        # )
        # intervals_img_pdf.savefig(fig_img)
        # plt.close(fig)
        # plt.close(fig_img)
        chunk_num += 1

    missing_periods.close()
    # intervals_ts_pdf.close()
    # intervals_img_pdf.close()
    labeled_dataset = pd.DataFrame(interval_labels)
    labeled_dataset.to_csv(
        f'st{instrument}_dataset_labels.txt',
        header=True,
        index=False
    )


def main(instrument='a'):
    data_file = (
        '/Users/ndmiles/ClassWork/FallQuarter2021/aos_c204/'
         f'aos_c204_final_project/data/st{instrument}_dataset_cleaned.txt'
    )
    icme_file = (
        '/Users/ndmiles/ClassWork/FallQuarter2021/aos_c204/'
        f'aos_c204_final_project/data/st{instrument}_icme_list.txt'
    )
    data_df = pd.read_csv(data_file, header=0, index_col=0, parse_dates=True)
    icme_df = pd.read_csv(icme_file, header=0, index_col=0, parse_dates=True)
    create_data_chunks(
        data_df=data_df,
        icme_df=icme_df,
        instrument=instrument,
        window_size=dt.timedelta(days=2)
    )


if __name__ == '__main__':
    main(instrument='a')
