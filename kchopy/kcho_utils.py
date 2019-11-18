import os
from os.path import isdir
import textwrap
from subprocess import check_call
import numpy as np
import nibabel as nb
import re
from pathlib import Path
from itertools import permutations
from tabulate import tabulate

from itertools import product
import inquirer


def search_and_select_one(name: str, location: str,
                          list_of_patterns: list, depth=1):
    """Search files with given patterns around the location and choose.

    Key arguments:
        title: str, name of the file that is being searched for.
        location: str, location of search.
        patterns: list, regrex patterns.
        depth: int, search parent depth, eg)0, 1, 2
    """
    # list of directories and serach patterns
    location = Path(location)

    # if file location is given as the `location`, set it as the parent
    # directory.
    if location.is_file():
        location = location.parent

    # search directory settings
    if depth == 1:
        list_search_directories = [location, location.parent]
    elif depth == 2:
        list_search_directories = [location, location.parent.parent]
    else:
        list_search_directories = [location]

    # list_of_patters = [
        # f'*all*_{self.modality}[_.]*nii.gz',
        # f'*{self.modality}*merged*.nii.gz'
        # ]

    # get combinations of the two lists
    list_of_dir_pat = list(product(
        list_search_directories,
        list_of_patterns))

    # search files
    matching_files = []
    for s_dir, pat in list_of_dir_pat:
        mf = list(Path(s_dir).glob(pat))
        if len(mf) > 0:
            matching_files += mf
        else:
            pass
    matching_files = list(set(matching_files))

    # check matching_files list
    if len(matching_files) == 1:
        final_file = matching_files[0]

    # if there are more than one merged skeleton files detected
    elif len(matching_files) > 1:
        questions = [
            inquirer.List(
                name,
                message="There are more than one matching {name}"
                        "Which is the correct merged file?",
                choices=matching_files,
                )
            ]
        answer = inquirer.prompt(questions)
        final_file = answer[name]
    # if no merged skeleton is detected
    else:
        final_file = 'missing'

    return final_file


def remove_overlapping_strings(str_list):
    perm = permutations(str_list, 2)

    str_list_new = []
    for list_two_strs in list(perm):
        a = list_two_strs[0]
        b = list_two_strs[1]

        res = ""
        non_overlap = []
        for num, i in enumerate(a):
            try:
                if i == b[num]:
                    res += i
                else:
                    res += '*'
            except:
                break

        first_slash_index = re.search('\/[^/]*\*', res).start()
        try:
            end_slash_index = re.search('\*[^/]*[\/$]', res).end()
            str_list_new.append(a[first_slash_index+1:end_slash_index-1])
        except:
            str_list_new.append(a[first_slash_index+1:])

    str_list_new = list(set(str_list_new))
    str_list_new.sort()

    return str_list_new


def with_suffix(nifti_p, suffix: str):
    '''stem nifti Path object name

    nifti_p: pathlib Path object
    '''

    if suffix.startswith('.'):
        return Path(str(nifti_p).split('.')[0] + suffix)
    else:
        return Path(str(nifti_p).split('.')[0] + '.' + suffix)


def nifti_stem(nifti_p):
    '''stem nifti Path object name

    nifti_p: pathlib Path object
    '''
    return str(nifti_p).split('.')[0]


def run(cmd):
    cmd_cleaned = re.sub(r'\s+', ' ', cmd)
    print("* {}".format(cmd_cleaned))
    check_call(cmd_cleaned, shell=True)


def run_job(cmd, queue, cores):
    cmd_cleaned = re.sub(r'\s+', ' ', cmd)
    cmd_server = 'bsub -q {} -n {} "{}"'.format(
        queue, cores, cmd_cleaned)
    print("* {}".format(cmd_server))
    print(cmd_server)
    check_call(cmd_server, shell=True)


def check_and_mkdirs(directory):
    if not isdir(directory):
        os.makedirs(directory)


def print_head(heading):
    print()
    print('-'*80)
    print(f'* {heading}')
    print('-'*80)


def print_df(df):
    """Print pandas dataframe using tabulate.

    Used to print outputs when the script is called from the shell
    Key arguments:
        df: pandas dataframe
    """
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print()


def presult(output):
    wrapper = textwrap.TextWrapper(initial_indent='\t',
                                   subsequent_indent='\t')
    print(wrapper.fill('{}'.format(output)))


def nhdr_write(nifti, bval, bvec, dwi_nhdr):
    pnlpipe = Path('/data/pnl/soft/pnlpipe3')
    converter_loc = pnlpipe / 'conversion/conversion'
    python_loc = pnlpipe / 'miniconda3/envs/pnlpipe3/bin'

    run('{}/python {}/nhdr_write.py \
        --nifti {} \
        --bval {} \
        --bvec {} \
        --nhdr {}'.format(
            python_loc, converter_loc, nifti, bval, bvec, dwi_nhdr))


def nhdr_write_non_dti(nifti_in, nhdr_out):
    pnlpipe = Path('/data/pnl/soft/pnlpipe3')
    converter_loc = pnlpipe / 'conversion/conversion'
    python_loc = pnlpipe / 'miniconda3/envs/pnlpipe3/bin'
    run('{}/python {}/nhdr_write.py --nifti {} --nhdr {}'.format(
        python_loc,
        converter_loc,
        nifti_in,
        nhdr_out))


def nifti_write(nrrd_or_nhdr, nii_out):
    python_loc = '/data/pnl/soft/pnlpipe3/miniconda3/envs/pnlpipe3/bin'
    converter_loc = '/data/pnl/soft/pnlpipe3/conversion/conversion'
    nii_prefix = re.sub('.nii.gz', '', str(nii_out))
    run('{}/python {}/nifti_write.py \
            -i {} -p {}'.format(python_loc,
                                converter_loc,
                                nrrd_or_nhdr,
                                nii_prefix))


def dim_corr(nii_img):
    run('fslswapdim {0} LR PA IS {0}'.format(nii_img))


def bet(img_in, img_out, f):
    run('bet {} {} -f {} -m'.format(img_in, img_out, f))


def extract_slice_from_nhdr(in_img, out_img, pos):
    run('unu slice -a 3 -p {} -i {} -o {}'.format(
        pos,
        in_img,
        out_img))


def mask_an_image(in_img, mask_img, out_img):
    run('unu 3op ifelse {} {} 0 | \
        unu save -f nrrd -e gzip -o {}'.format(
        mask_img,
        in_img,
        out_img))


def nhdr(prefix):
    if '.' in prefix:
        prefix = prefix.split('.')[0]
    return prefix+'.nhdr'


def nrrd(prefix):
    if '.' in prefix:
        prefix = prefix.split('.')[0]
    return prefix+'.nrrd'


def nii(prefix):
    if '.' in prefix:
        prefix = prefix.split('.')[0]
    return prefix+'.nii.gz'


def pick_darkest_label_from_kmeans_labels(kmeans_array, bg_data):
    labels = np.unique(kmeans_array[np.nonzero(kmeans_array)])

    # label_with_max_intensity = ''
    min_intensity = 100000000
    for label_number in labels:
        intensity = bg_data[np.where(kmeans_array == label_number)].mean()

        if intensity < min_intensity:
            ventricle_label = label_number
            min_intensity = intensity
        else:
            pass

    return ventricle_label


def pick_brightest_label_from_kmeans_labels(kmeans_array, bg_data):
    labels = np.unique(kmeans_array[np.nonzero(kmeans_array)])

    # label_with_max_intensity = ''
    max_intensity = 0
    for label_number in labels:
        intensity = bg_data[np.where(kmeans_array == label_number)].mean()

        if intensity > max_intensity:
            ventricle_label = label_number
            max_intensity = intensity
        else:
            pass

    return ventricle_label


def rms(data1, data2):
    return np.sqrt(np.array(data1 - data2)**2)


def rms_between_all(data, mask):
    '''
    Estimate difference between each volume to all other volums
    where data is 4D matrix
    '''

    print(data.shape)
    vol_num_array = np.arange(data.shape[-1])
    diff_array = np.zeros((data.shape[-1], data.shape[-1]))

    for vol_num in vol_num_array:
        print(vol_num)
        data_1 = data[:, :, :, vol_num]
        for other_vol_num in vol_num_array:
            # if the same image
            if vol_num == vol_num_array:
                diff_array[vol_num, vol_num_array] = 0
            else:
                data_2 = data[:, :, :, other_vol_num]
                # diff
                diff = rms(data_1, data_2)
                diff_mean = diff.mean()
                diff_array[vol_num, vol_num_array] = diff_mean

    return diff_array


def compare_volume(data1, data2, volume):
    return rms(data1[:, :, :, volume], data2[:, :, :, volume])


def compare_to_first_volume(data, volume):
    return rms(data[:, :, :, 0], data[:, :, :, volume])


def compare_to_first_volume_all(data):
    volumes = data.shape[-1]

    rms_means = []
    for volume_num in np.arange(volumes):
        rms_means.append(compare_to_first_volume(
            data,
            volume_num
        ).mean())

    return rms_means


def extract_b0_and_average(dwi, bval, b0_avg_out):
    '''
    Extract B0 according to the bval text file (fsl format)
    then saves average
    '''
    dwi_img = nb.load(dwi)
    dwi_data = dwi_img.get_data()

    bval_array = np.loadtxt(bval)
    b0_points = np.where(bval_array == 0)[0]

    b0_data = dwi_data[:, :, :, np.where(b0_points == 0)[0]]
    b0_data_avg = np.mean(b0_data, axis=3)
    nb.Nifti1Image(b0_data_avg,
                   affine=dwi_img.affine).to_filename(b0_avg_out)


def extract_b0_and_average_nibabel(data, bval):
    ''' Extract B0 according to the bval text file (fsl format)
    '''
    # b0_points = np.where(bval==0)[0]
    b0_data = data[:, :, :, np.where(bval == 0)[0]]
    b0_data_avg = np.mean(b0_data, axis=3)
    return b0_data_avg


def extract_average_from_shell(data, bval, target_bvalue):
    '''Extract B0 according to the bval text file (fsl format)
    '''
    b_data = data[:, :, :, np.where(bval == target_bvalue)[0]]
    b_data_avg = np.mean(b_data, axis=3)
    return b_data_avg
