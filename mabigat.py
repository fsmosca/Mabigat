#!/usr/bin/env python


"""Mabigat NNUE hyperparameter optimizer

Creates NNUE net by trying to optimize the parameters that are involve in the generation
of training and validation positions as well as learning. It uses the Optuna framework to
optimize those parameters."""


__author__ = 'fsmosca'
__script_name__ = 'mabigat'
__description__ = 'Mabigat NNUE net hyperparameter optimizer'
__version__ = 'v0.7.0'
__credits__ = ['musketeerchess']


import sys
import subprocess
from pathlib import Path
import shutil
import logging
import argparse
import configparser
import ast
import copy
import time

import optuna
from plotly.subplots import make_subplots
import plotly.graph_objects as go


logger = logging.getLogger('mabigat')


PLOT_WIDTH = 1000
PLOT_HEIGHT_CONTOUR = 1000
PLOT_HEIGHT = 450
OPTUNA_PLOT_BACKGROUND = '#F7D0CA'


is_panda_ok = True
try:
    import pandas as pd
    pd.options.plotting.backend = "plotly"
except ModuleNotFoundError:
    is_panda_ok = False
    print('Warning! pandas is not installed.')


LEARNING_FLAG_PARAMS = [
    'set_recommended_uci_options', 'save_only_once',
    'no_shuffle', 'assume_quiet', 'smart_fen_skipping',
    'smart_fen_skipping_for_validation'
]


class TrainingSFNNUE:
    def __init__(self, enginefn, engine_options, ini_file,
                 sub_study_folder='log', eval_save_dir='evalsave'):
        self.enginefn = enginefn
        self.engine_options = engine_options
        self.ini_file = ini_file
        self.sub_study_folder = sub_study_folder
        self.eval_save_dir = eval_save_dir
        self.engine_option_names = self.get_engine_option_names()
        self.training_pos = get_num_positions(ini_file, mode='train')
        self.validation_pos = get_validation_count(ini_file)

    def send(self, proc, command):
        proc.stdin.write(f'{command}\n')

    def generate_positions(
            self,
            num_trials,
            mode,
            study_name,
            output_fn,
            generation_param,
            generation_param_to_optimze
    ):
        eng = subprocess.Popen(self.enginefn, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)

        self.send(eng, 'uci')

        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'uciok' in line:
                break

        # Set options
        self.send(eng, f'setoption name Debug Log File value {self.sub_study_folder}/{mode}_{study_name}_trial_{num_trials}_sflog.txt')
        for n in self.engine_options:
            for k, v in n.items():
                if k.lower() == 'debug log file' or k.lower() == 'engine_file':
                    continue
                self.send(eng, f'setoption name {k} value {v}')

        # Find an engine option names that are included in generation_param.
        # We don't include them in gensfen command.
        # Also save the dict that are engine options.
        excluded, eng_opt = [], []
        for n in generation_param:
            for k, v in n.items():
                if k.lower() in self.engine_option_names:
                    excluded.append(k.lower())  # list
                    eng_opt.append(n)  # list of dict
                    break

        # Send the latest engine options.
        if len(eng_opt):
            for n in eng_opt:
                for k, v in n.items():
                    self.send(eng, f'setoption name {k} value {v}')

        # No engine option names should be in generation_param_to_optimze.

        self.send(eng, 'isready')
        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'readyok' in line:
                break

        self.send(eng, 'ucinewgame')
        self.send(eng, 'isready')
        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'readyok' in line:
                break

        # Build the command line to generate the positions.
        cmd = f'gensfen output_file_name {output_fn}'

        # Add param to be optimized.
        for par in generation_param_to_optimze:
            for k, v in par.items():
                kval = k.lower()
                if kval == 'set_recommended_uci_options':
                    cmd += f' {k}'
                elif kval == 'ensure_quiet':
                    cmd += f' {k}'
                elif kval == 'num_pos':
                    cmd += f' loop {v}'
                elif kval in excluded:
                    continue
                else:
                    cmd += f' {k} {v}'

        # Add the param that is not to be optimized.
        for par in generation_param:
            for k, v in par.items():
                kval = k.lower()
                if kval == 'set_recommended_uci_options':
                    cmd += f' {k}'
                elif kval == 'ensure_quiet':
                    cmd += f' {k}'
                elif kval == 'num_pos':
                    cmd += f' loop {v}'
                elif kval in excluded:
                    continue
                else:
                    cmd += f' {k} {v}'

        logger.debug(f'command line: {cmd}')

        # Execute the command line.
        self.send(eng, f'{cmd}')

        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'gensfen finished' in line.lower():
                break

        logger.info(f'done {mode} data generation')

        self.send(eng, 'quit')

    def learn(
            self,
            num_trials,
            study_name,
            target_dir,
            validation_set_file_name,
            learning_param,
            learning_param_to_optimize
    ):
        eng = subprocess.Popen(self.enginefn, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)

        self.send(eng, 'uci')

        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'uciok' in line:
                break

        # Set options
        self.send(eng, f'setoption name Debug Log File value {self.sub_study_folder}/learn_{study_name}_trial_{num_trials}_sflog.txt')
        self.send(eng, f'setoption name EvalSaveDir value {self.eval_save_dir}')
        for n in self.engine_options:
            for k, v in n.items():
                if k.lower() == 'debug log file' or k.lower() == 'engine_file':
                    continue
                self.send(eng, f'setoption name {k} value {v}')

        # Find an engine option names that are included in learning_param.
        # We don't include them in learn command.
        # Also save the dict that are engine options.
        excluded, eng_opt = [], []
        for n in learning_param:
            for k, v in n.items():
                if k in self.engine_option_names:
                    excluded.append(k)  # list
                    eng_opt.append(n)  # list of dict
                    break

        # Send the latest engine options.
        if len(eng_opt):
            for n in eng_opt:
                for k, v in n.items():
                    self.send(eng, f'setoption name {k} value {v}')

        self.send(eng, 'isready')
        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'readyok' in line:
                break

        self.send(eng, 'ucinewgame')
        self.send(eng, 'isready')

        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'readyok' in line:
                break

        cmd = f'learn targetdir {target_dir}'
        cmd += f' validation_set_file_name {validation_set_file_name}'

        # Set param to be optimized.
        for par in learning_param_to_optimize:
            for k, v in par.items():
                if k in LEARNING_FLAG_PARAMS:
                    if v == '1':
                        cmd += f' {k}'
                elif k in excluded:
                    continue
                else:
                    cmd += f' {k} {v}'

        # Set param not to be optimized.
        for par in learning_param:
            for k, v in par.items():
                if k in LEARNING_FLAG_PARAMS:
                    if v == '1':
                        cmd += f' {k}'
                elif k in excluded:
                    continue
                else:
                    cmd += f' {k} {v}'

        logger.debug(f'command line: {cmd}')

        self.send(eng, f'{cmd}')

        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            if 'finished saving evaluation file' in line.lower() and '/final' in line.lower():
                break
            else:
                if 'val_loss' in line:
                    self.plot_engine_learning(study_name, num_trials)

        self.send(eng, 'quit')

        logger.info('done learning')

    def plot_engine_learning(self, study_name, num_trials):
        """
        Plot data from learning like , val and train loses.
        """
        try:
            sflog = f'{self.sub_study_folder}/learn_{study_name}_trial_{num_trials}_sflog.txt'
            val_loss, train_loss, sfens, epochs, lr, move_acc = plot_val_train_loss(sflog)

            # Attempt to plot if there values in val and train losses.
            if len(val_loss) and len(train_loss):
                fig = make_subplots(rows=2, cols=2, vertical_spacing=0.18,
                                    subplot_titles=('val loss', 'train loss', 'move accuracy', 'learning rate'))

                fig.add_trace(
                    go.Scatter(x=epochs, y=val_loss, name='val loss'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=epochs, y=train_loss, name='train loss'),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Scatter(x=epochs, y=move_acc, name='move accuracy'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=epochs, y=lr, name='learning rate'),
                    row=2, col=2
                )

                fig['layout']['xaxis1']['title'] = 'epoch'
                fig['layout']['xaxis2']['title'] = 'epoch'
                fig['layout']['xaxis3']['title'] = 'epoch'
                fig['layout']['xaxis4']['title'] = 'epoch'

                fig['layout']['yaxis1']['title'] = 'loss'
                fig['layout']['yaxis2']['title'] = 'loss'
                fig['layout']['yaxis3']['title'] = 'move accuracy %'
                fig['layout']['yaxis4']['title'] = 'learning rate'

                fig.update_layout(height=800, width=1500,
                                  title_text=f"Learning, trainpos: {self.training_pos}, valpos: {self.validation_pos}",
                                  legend_title="Legend")

                include_plotlyjs = True
                full_html, auto_play = False, False
                with open(f'{self.sub_study_folder}/{study_name}_trial_{num_trials}_learning_plot.html', 'w') as f:
                    f.write(fig.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, auto_play=auto_play))

        except Exception as err:
            logger.warning(f'warning in plotting val_loss and val_train as {err}')

    def get_engine_option_names(self):
        option_names = []
        eng = subprocess.Popen(self.enginefn, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
        self.send(eng, 'uci')

        for eline in iter(eng.stdout.readline, ''):
            line = eline.strip()
            # option name Debug Log File type string default
            if line.startswith('option name'):
                value = line.split('option name ')[1].split(' type')[0]
                option_names.append(value.lower())
            if 'uciok' in line:
                break

        self.send(eng, 'quit')

        return option_names


def delete_folder(folder: str):
    folder_path = Path(folder)
    if folder_path.is_dir():
        shutil.rmtree(folder_path)


def create_folder(folder):
    new_folder = Path(folder)
    new_folder.mkdir(exist_ok=True)


def move_data(src, dst):
    shutil.move(src, dst)


def get_sampler(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('OPTUNA'))
    return data.get('sampler', 'tpe')


def get_study_name(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('OPTUNA'))
    return data.get('study_name', None)


def get_study_num_trials(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('OPTUNA'))
    return int(data.get('num_trials', 100))


def get_engine_file(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('ENGINE'))
    return data.get('engine_file', None)


def get_plot_params(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('PLOT'))
    opt_value = data.get('plot_params', None)
    return ast.literal_eval(opt_value)


def get_engine_threads(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('ENGINE'))
    return int(data.get('threads', 1))


def get_engine_options(ini_file):
    options = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'ENGINE':
                options.append({opt_name: opt_value})

    return options


def get_engine_hash_mb(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('ENGINE'))
    return int(data.get('hash', 128))


def get_depth(ini_file, mode='train'):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    if mode.lower() == 'val':
        data = dict(parser.items('VALIDATION_POS_GENERATION'))
        return int(data.get('depth', 0))
    else:
        data = dict(parser.items('TRAINING_POS_GENERATION'))
        return int(data.get('depth', 3))


def get_num_positions(ini_file, mode='train'):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    if mode.lower() == 'val':
        data = dict(parser.items('VALIDATION_POS_GENERATION'))
        return int(data.get('num_pos', 0))
    else:
        data = dict(parser.items('TRAINING_POS_GENERATION'))
        return int(data.get('num_pos', 8000000000))


def get_validation_count(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('LEARNING'))
    return int(data.get('validation_count', 2000))


def get_book(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('TRAINING_POS_GENERATION'))
    return data.get('book', None)


def get_cutechess_rounds(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return int(data.get('rounds', 50))


def get_cutechess_concurrency(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return int(data.get('concurrency', 1))


def get_cutechess_draw(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return data.get('draw', None)


def get_cutechess_resign(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return data.get('resign', None)


def get_python_file(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return data.get('python_file', 'python')


def get_cutechess_book(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return data.get('book', '')


def get_eval_save_interval(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('LEARNING'))
    return int(data.get('eval_save_interval', 100000000))


def get_loss_output_interval(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('LEARNING'))
    return int(data.get('loss_output_interval', 1000000))


def get_use_best_param(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('MABIGAT'))
    return int(data.get('use_best_param', 1))


def get_init_best_match_result(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('MABIGAT'))
    return float(data.get('init_best_match_result', 0.5))


def get_cutechess_cli_path(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return data.get('cutechess_cli_path', './cutechess/cutechess-cli.exe')


def get_cutechess_time_control(ini_file):
    parser = configparser.ConfigParser()
    parser.read(ini_file)
    data = dict(parser.items('CUTECHESS'))
    return data.get('time_control', '0/2+0.05')


def get_training_gen_param_to_optimize(ini_file, trial):
    """
    Asks the optimizer the param values to try.
    """
    training_gen_param_to_optimize = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'TRAINING_POS_GENERATION_PARAM_TO_OPTIMIZE':
                # categorical variable
                if '[' in opt_value and ']' in opt_value:
                    n_value = ast.literal_eval(opt_value)
                    var = trial.suggest_categorical(opt_name, n_value)
                    training_gen_param_to_optimize.append({opt_name: var})

                # continuous variable
                elif '(' in opt_value and ')' in opt_value:
                    n_value = ast.literal_eval(opt_value)

                    if isinstance(n_value[0], float):
                        if len(n_value) == 3:
                            (min, high, step) = n_value
                            var = trial.suggest_float(opt_name, min, high, step=step)
                            training_gen_param_to_optimize.append({opt_name: var})
                        elif len(n_value) == 2:
                            (min, high) = n_value
                            var = trial.suggest_float(opt_name, min, high)
                            training_gen_param_to_optimize.append({opt_name: var})

                    elif isinstance(n_value[0], int):
                        if len(n_value) == 3:
                            (min, high, step) = n_value
                            var = trial.suggest_int(opt_name, min, high, step)
                            training_gen_param_to_optimize.append({opt_name: var})
                        elif len(n_value) == 2:
                            (min, high) = n_value
                            var = trial.suggest_int(opt_name, min, high)
                            training_gen_param_to_optimize.append({opt_name: var})

    return training_gen_param_to_optimize


def get_validation_gen_param_to_optimize(ini_file, trial):
    """
    Asks the optimizer the param values to try.
    """
    validation_gen_param_to_optimize = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'VALIDATION_POS_GENERATION_PARAM_TO_OPTIMIZE':
                # categorical variable
                if '[' in opt_value and ']' in opt_value:
                    n_value = ast.literal_eval(opt_value)
                    var = trial.suggest_categorical(opt_name, n_value)
                    validation_gen_param_to_optimize.append({opt_name: var})

                # continuous variable
                elif '(' in opt_value and ')' in opt_value:
                    n_value = ast.literal_eval(opt_value)

                    if isinstance(n_value[0], float):
                        if len(n_value) == 3:
                            (min, high, step) = n_value
                            var = trial.suggest_float(opt_name, min, high, step=step)
                            validation_gen_param_to_optimize.append({opt_name: var})
                        elif len(n_value) == 2:
                            (min, high) = n_value
                            var = trial.suggest_float(opt_name, min, high)
                            validation_gen_param_to_optimize.append({opt_name: var})

                    elif isinstance(n_value[0], int):
                        if len(n_value) == 3:
                            (min, high, step) = n_value
                            var = trial.suggest_int(opt_name, min, high, step)
                            validation_gen_param_to_optimize.append({opt_name: var})
                        elif len(n_value) == 2:
                            (min, high) = n_value
                            var = trial.suggest_int(opt_name, min, high)
                            validation_gen_param_to_optimize.append({opt_name: var})

    return validation_gen_param_to_optimize


def get_training_gen_param(ini_file):
    """
    These params are not to be optimized but changed by user probably to
    be different from its default values.
    """
    param = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'TRAINING_POS_GENERATION':
                param.append({opt_name: opt_value})

    return param


def get_validation_gen_param(ini_file):
    """
    These params are not to be optimized but changed by user probably to
    be different from its default values.
    """
    param = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'VALIDATION_POS_GENERATION':
                param.append({opt_name: opt_value})

    return param


def get_learning_param_to_optimize(ini_file, trial):
    """
    Asks the optimizer the param values to try.
    """
    param = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'LEARNING_PARAM_TO_OPTIMIZE':
                # categorical variable
                if '[' in opt_value and ']' in opt_value:
                    n_value = ast.literal_eval(opt_value)
                    var = trial.suggest_categorical(opt_name, n_value)
                    param.append({opt_name: var})

                # continuous variable
                elif '(' in opt_value and ')' in opt_value:
                    n_value = ast.literal_eval(opt_value)

                    if isinstance(n_value[0], float):
                        if len(n_value) == 3:
                            (min, high, step) = n_value
                            var = trial.suggest_float(opt_name, min, high, step=step)
                            param.append({opt_name: var})
                        elif len(n_value) == 2:
                            (min, high) = n_value
                            var = trial.suggest_float(opt_name, min, high)
                            param.append({opt_name: var})

                    elif isinstance(n_value[0], int):
                        if len(n_value) == 3:
                            (min, high, step) = n_value
                            var = trial.suggest_int(opt_name, min, high, step)
                            param.append({opt_name: var})
                        elif len(n_value) == 2:
                            (min, high) = n_value
                            var = trial.suggest_int(opt_name, min, high)
                            param.append({opt_name: var})

    return param


def get_learning_param(ini_file):
    """
    These params are not to be optimized but changed by user probably to
    be different from its default values.
    """
    param = []

    parser = configparser.ConfigParser()
    parser.read(ini_file)

    for section_name in parser.sections():
        for opt_name, opt_value in parser.items(section_name):
            if section_name == 'LEARNING':
                param.append({opt_name: opt_value})

    return param


def plot_val_train_loss(sflog):
    """
    Always read the sflog from the start.
    """
    val_loss, train_loss, sfens, epochs, lr, move_acc = [], [], [], [], [], []

    with open(sflog) as f:
        for lines in f:
            line = lines.rstrip()

            # PROGRESS (calc_loss): Sun Mar 28 01:20:13 2021, 1000000 sfens, 37313 sfens/second, epoch 1
            if 'PROGRESS' in line:
                # Get sfens
                value = int(line.split(' sfens')[0].split(', ')[1])
                sfens.append(value)

                # Get epoch
                value = int(line.split('epoch ')[1])
                epochs.append(value)

            # - learning rate = 1
            elif 'learning rate = ' in line:
                value = float(line.split('learning rate = ')[1])
                lr.append(value)

            # val_loss       = 0.0782639
            elif 'val_loss' in line:
                value = float(line.split('= ')[1])
                val_loss.append(value)

            # train_loss = 0.201731
            elif 'train_loss' in line:
                value = float(line.split('= ')[1])
                train_loss.append(value)

            # - move accuracy = 0.4875%
            elif 'move accuracy = ' in line and not 'random move accuracy = ' in line:
                value = float(line.split('move accuracy = ')[1].split('%')[0])
                move_acc.append(value)

    # In the beginning, there is no train loss, we will insert a value from 2nd epoch.
    if len(val_loss) - len(train_loss) == 1:
        if len(train_loss):
            value_to_insert = train_loss[0]
            train_loss.insert(0, value_to_insert)

    # logger.debug(f'val_loss: {val_loss}')
    # logger.debug(f'train_loss: {train_loss}')

    return val_loss, train_loss, sfens, epochs, lr, move_acc


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog='%s %s' % (__script_name__, __version__),
        description=f'{__description__}',
        epilog='%(prog)s')
    parser.add_argument('--ini-file', required=True,
                        help='The path/file or file of initialization file. Example:\n'
                             'python mabigat.py --ini-file ./ini/example.ini')

    args = parser.parse_args()

    # Get ini file
    ini_file = args.ini_file
    ini_file_path = Path(ini_file)
    if not ini_file_path.is_file():
        raise Exception(f'ini file {ini_file} does not exists, usage: mabigat.py --ini-file <your ini file>')

    cwd = Path.cwd().as_posix()

    # Get study name.
    study_name = get_study_name(ini_file)
    if study_name is None:
        print('Error, define study_name in .ini file! under OPTUNA section')
        raise

    study_folder = Path(cwd, 'study')
    create_folder(study_folder)

    sub_study_folder = Path(study_folder, study_name)
    create_folder(sub_study_folder)

    eval_save_folder = Path(sub_study_folder, 'evalsave')

    log_filename = f'{sub_study_folder}/{study_name}_log.txt'

    # Define logger in detail after we get the study name.
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename=log_filename, mode='a')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fh_formatter = logging.Formatter('%(message)s')
    ch_formatter = logging.Formatter('%(message)s')
    fh.setFormatter(fh_formatter)
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # --- mabigat ---
    use_best_param = get_use_best_param(ini_file)
    init_best_match_result = get_init_best_match_result(ini_file)

    # --- optuna ---
    n_trials = get_study_num_trials(ini_file)

    # --- engine ---
    engine_file = get_engine_file(ini_file)
    if engine_file is None:
        logger.exception('Error, define engine_file in mabigat.ini file! under ENGINE section')
        logger.exception('engine_file = path/toyourenginefile/eng.exe')
        raise
    threads = get_engine_threads(ini_file)
    hash_mb = get_engine_hash_mb(ini_file)

    # --- training / validation ---
    train_depth = get_depth(ini_file)
    val_depth = get_depth(ini_file, 'val')
    numpos_train = get_num_positions(ini_file)
    numpos_val = get_num_positions(ini_file, 'val')
    book = get_book(ini_file)

    # --- learning ---
    eval_save_interval = get_eval_save_interval(ini_file)
    loss_output_interval = get_loss_output_interval(ini_file)

    # Define class where pos generation and learning methods are called.
    engine_options = get_engine_options(ini_file)
    nnue = TrainingSFNNUE(engine_file, engine_options, ini_file,
                          sub_study_folder=sub_study_folder,
                          eval_save_dir=eval_save_folder)

    # Define storage, sampler and study.
    storage_name = f'sqlite:///{sub_study_folder}/{study_name}.db'

    sampler_name = get_sampler(ini_file)
    if sampler_name.lower() == 'cmaes':
        # Avoid using categorical pram type.
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html
        sampler = optuna.samplers.CmaEsSampler(seed=100)
    else:
        # tpe
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html
        sampler = optuna.samplers.TPESampler(seed=100, multivariate=False)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True,
        sampler=sampler
    )

    # Logging to file and console.
    logger.info(f'Mabigat {__version__}')
    logger.info(f'optuna {optuna.__version__}\n')

    logger.info(f'engine   : {engine_file}')
    logger.info(f'threads  : {threads}')
    logger.info(f'hash     : {hash_mb}\n')

    logger.info(f'study name        : {study_name}')
    logger.info(f'sampler/optimizer : {sampler_name}')
    logger.info(f'number of trials  : {n_trials}\n')

    logger.info(f'number of training positions  : {numpos_train}')

    if numpos_val == 0:
        logger.info('number of validation positions: for optimization')
    else:
        logger.info(f'number of validation positions: {numpos_val}')

    logger.info(f'training depth                : {train_depth}')

    if val_depth == 0:
        logger.info('validation depth              : for optimization')
    else:
        logger.info(f'validation depth              : {val_depth}')

    logger.info(f'book                          : {book}\n')

    logger.info(f'eval_save_interval  : {eval_save_interval}')
    logger.info(f'loss_output_interval: {loss_output_interval}\n')

    # Start the optimization.
    for _ in range(n_trials):

        num_trials = len(study.trials)
        logger.info(f'starting trial: {num_trials}')

        trial = study.ask()

        # 2. Generate training positions
        # Manage folders and files.
        positions, depth = numpos_train, train_depth
        mode = 'train'
        train_folder = f'{sub_study_folder}/train'

        delete_folder(train_folder)
        create_folder(train_folder)
        train_nn_output_path_file = f'{train_folder}/{study_name}_training_trial_{num_trials}_pos_{positions}_depth_{depth}.binpack'
        train_nn_output_file = f'{study_name}_training_trial_{num_trials}_pos_{positions}_depth_{depth}.binpack'

        # Get the params that are not to be optimized.
        training_gen_param = get_training_gen_param(ini_file)

        # Get the params that are to be optimized.
        training_gen_param_to_optimize = get_training_gen_param_to_optimize(ini_file, trial)
        if len(training_gen_param_to_optimize):
            logger.debug(f'Training pos generation param to optimize:')
            for n in training_gen_param_to_optimize:
                logger.debug(n)

        logger.info('generating training positions ...')

        # Generate positions.
        nnue.generate_positions(
            num_trials,
            mode,
            study_name,
            train_nn_output_path_file,
            training_gen_param,
            training_gen_param_to_optimize
        )


        # 3. Generate validation positions
        # Manage folders and files.
        positions, depth = numpos_val, val_depth
        mode = 'val'
        val_folder = f'{sub_study_folder}/val'

        delete_folder(val_folder)
        create_folder(val_folder)

        # Get the params that are not to be optimized.
        validation_gen_param = get_validation_gen_param(ini_file)

        # Add training param.
        for n in training_gen_param:
            for k, v in n.items():
                if k != 'num_pos' and k != 'depth':
                    validation_gen_param.append(n)

        # Get the params that are to be optimized.
        validation_gen_param_to_optimize = get_validation_gen_param_to_optimize(ini_file, trial)
        if len(validation_gen_param_to_optimize) == 0:
            validation_gen_param_to_optimize = copy.copy(training_gen_param_to_optimize)

        if len(validation_gen_param_to_optimize):
            logger.debug(f'validation pos generation param to optimize:')
            for n in validation_gen_param_to_optimize:
                logger.debug(n)

        # If depth is to be optimized.
        if depth == 0:
            found = False
            for n in validation_gen_param_to_optimize:
                for k, v in n.items():
                    if k == 'depth':
                        found = True
                        depth = v
                        break
                if found:
                    break

        val_nn_output_file = f'{study_name}_validation_trial_{num_trials}_pos_{positions}_depth_{depth}.binpack'
        val_nn_output_path_file = f'{val_folder}/{val_nn_output_file}'

        logger.info('generating validation positions ...')

        # Generate positions.
        nnue.generate_positions(
            num_trials,
            mode,
            study_name,
            val_nn_output_path_file,
            validation_gen_param,
            validation_gen_param_to_optimize
        )

        # 4. Learning

        delete_folder(eval_save_folder)
        create_folder(eval_save_folder)
        bins_folder = f'{sub_study_folder}/{study_name}_net_bins'
        create_folder(bins_folder)
        targetdir = train_folder
        learning_param = get_learning_param(ini_file)
        learning_param_to_optimize = get_learning_param_to_optimize(ini_file, trial)

        if len(learning_param_to_optimize):
            logger.debug(f'Learning param to optimize:')
            for n in learning_param_to_optimize:
                logger.debug(n)

        logger.info('run learning ...')

        nnue.learn(
            num_trials,
            study_name,
            targetdir,
            val_nn_output_path_file,
            learning_param,
            learning_param_to_optimize
        )

        # Backup bins after learning is done.
        time.sleep(3)
        move_data(f'{eval_save_folder}/final/nn.bin', f'{bins_folder}/{num_trials}_nn.bin')

        # Backup train and val bins
        time.sleep(3)
        backup_folder = f'{sub_study_folder}/{study_name}_train_and_val_bins'
        create_folder(backup_folder)

        # Backup the training file.
        time.sleep(3)
        move_data(train_nn_output_path_file, f'{backup_folder}/{train_nn_output_file}')

        # Backup the validation file.
        time.sleep(3)
        move_data(val_nn_output_path_file, f'{backup_folder}/{val_nn_output_file}')

        # Cleanup
        time.sleep(3)
        delete_folder(train_folder)
        delete_folder(val_folder)

        # 5. Create match to test the nn output.
        time.sleep(3)
        reported_match_result = init_best_match_result

        python_file = get_python_file(ini_file)
        rounds = get_cutechess_rounds(ini_file)
        cutechess_cli_path = get_cutechess_cli_path(ini_file)
        time_control = get_cutechess_time_control(ini_file)
        book = get_cutechess_book(ini_file)
        concurrency = get_cutechess_concurrency(ini_file)
        draw = get_cutechess_draw(ini_file)
        resign = get_cutechess_resign(ini_file)

        match_result, pruned_trial = None, False

        if num_trials >= 1:
            tour_start = time.perf_counter()
            opt1_1 = f'name={num_trials}_nn'
            nn_path = Path(cwd, f'{bins_folder}/{num_trials}_nn.bin')
            opt1_2 = f'option.EvalFile={nn_path}'

            best_trial_value = study.best_trial.values  # a list

            if use_best_param:
                best_trial_num = study.best_trial.number
                opt2_1 = f'name={best_trial_num}_nn'
                nn_path = Path(cwd, f'{bins_folder}/{best_trial_num}_nn.bin')
                opt2_2 = f'option.EvalFile={nn_path}'
            else:
                best_trial_num = 0
                opt2_1 = f'name={best_trial_num}_nn'
                nn_path = Path(cwd, f'{bins_folder}/{best_trial_num}_nn.bin')
                opt2_2 = f'option.EvalFile={nn_path}'

            logger.info(f'Execute engine vs engine match between {num_trials}_nn.bin and {best_trial_num}_nn.bin')

            cmd = f'{python_file} match.py {sub_study_folder} {study_name} {cutechess_cli_path} ' \
                  f'{engine_file} {opt1_1} {opt1_2} {opt2_1} {opt2_2} {rounds} {time_control} ' \
                  f'{book} {concurrency} {draw} {resign}'
            logger.debug(f'cmd: {cmd}')

            match = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            for eline in iter(match.stdout.readline, ''):
                line = eline.strip()
                if 'result ' in line:
                    match_result = float(line.split('result ')[1])
                    break

            logger.debug(f'tour elapse (s): {time.perf_counter() - tour_start: 0.1f}, games: {rounds*2}')

            # If match result is broken, we continue the study but prune this trial.
            if match_result is None:
                logger.exception('There is error in the match, prune this trial.')
                pruned_trial = True
            else:
                logger.info(f'Match done! actual result: {match_result}, point of view: {opt1_1}')

                if use_best_param:
                    # If we use the best param so far against the suggested values from the optimizer
                    # we need to adjust the result value reported to optimizer.
                    if match_result > init_best_match_result:
                        result_diff = match_result - init_best_match_result
                        current_best_trial_value = float(best_trial_value[0])  # get index 0 for single objective
                        reported_match_result = current_best_trial_value + result_diff

                        # The adjusted result may exceed 1.0 or 100%. This is ok as we are maximizing the result.
                        logger.info(f'use_best_param: {use_best_param}, adjusted result: {reported_match_result}')
                    else:
                        reported_match_result = match_result

        if pruned_trial:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            raise
        else:
            study.tell(trial, reported_match_result)

        logger.debug(f'best trial {study.best_trial.number}')
        logger.debug(f'best value {study.best_value}')
        logger.debug(f'best param {study.best_params}')

        # Cleanup eval save folder.
        try:
            delete_folder(eval_save_folder)
        except PermissionError:
            # We delete this file later before we enter learning next time.
            logger.debug(f'PermissionError, we delete this file later before learning.')
        except Exception as err:
            logger.exception(f'Unexpected error in deleting eval save folder as {err}')
            raise

        # Build pandas dataframe, and save to csv file.
        if is_panda_ok:
            df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
            logger.info(f'{df.to_string(index=False)}\n')
            df.to_csv(f'{sub_study_folder}/{study_name}.csv', index=False)
        else:
            for n in study.trials:
                logger.info(f'trial: {n.number}, params: {n.params}, objective value: {n.value}')

        # Plot optuna visualization features.
        try:
            # Print plot in png and html every 4 trials.
            if num_trials >= 2:
                params_to_plot = get_plot_params(ini_file)
                logger.debug(f'params to plot: {params_to_plot}')

                # history
                fig0 = optuna.visualization.plot_optimization_history(study)
                fig0.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT)
                fig0.update_layout(paper_bgcolor=OPTUNA_PLOT_BACKGROUND)

                # contour
                fig1 = optuna.visualization.plot_contour(study, params=params_to_plot)
                fig1.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT_CONTOUR)
                fig1.update_layout(paper_bgcolor=OPTUNA_PLOT_BACKGROUND)

                # slice
                fig2 = optuna.visualization.plot_slice(study, params=params_to_plot)
                fig2.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT)
                fig2.update_layout(paper_bgcolor=OPTUNA_PLOT_BACKGROUND)

                # importances
                fig3 = optuna.visualization.plot_param_importances(study)
                fig3.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT)
                fig3.update_layout(paper_bgcolor=OPTUNA_PLOT_BACKGROUND)

                # Save to single html.
                # include_plotlyjs=True, output file is bigger, but internet is not required.
                # include_plotlyjs='cdn', output file is smaller, but internet is required.
                include_plotlyjs = True
                full_html, auto_play = False, False
                with open(f'{sub_study_folder}/{study_name}_optimizer_plot.html', 'w') as f:
                    f.write(fig0.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, auto_play=auto_play))
                    f.write(fig1.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, auto_play=auto_play))
                    f.write(fig2.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, auto_play=auto_play))
                    f.write(fig3.to_html(full_html=full_html, include_plotlyjs=include_plotlyjs, auto_play=auto_play))

        except Exception as err:
            logger.debug(f'plotting error, as {err}')

    logger.info('optimization done')


if __name__ == "__main__":
    main()
