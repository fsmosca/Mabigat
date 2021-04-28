#!/usr/bin/python

"""
This is based on clop-cutechess-cli.py from https://github.com/cutechess/cutechess.
"""

from subprocess import Popen, PIPE
import sys
import logging
from pathlib import Path


games = 2

def main(argv=None):
    sub_study_folder = argv[0]
    study_name = argv[1]
    cutechess_cli_path = argv[2]
    engine = argv[3]
    fcp = f'cmd={engine} {argv[4]} {argv[5]}'
    scp = f'cmd={engine} {argv[6]} {argv[7]}'
    rounds = argv[8]
    tc = argv[9]
    book = argv[10]
    concurrency = argv[11]
    draw = argv[12]
    resign = argv[13]

    logging.basicConfig(
        filename=f'{sub_study_folder}/cutechess_log.txt',
        filemode='a',
        level=logging.DEBUG,
        format='%(message)s'
    )

    cutechess_cli_path = Path(cutechess_cli_path).resolve()

    options = f'-concurrency {concurrency}'
    options += ' -recover'
    options += f' -rounds {rounds} -games {games} -repeat'
    options += f' -each {tc} proto=uci'
    options += f' -pgnout {sub_study_folder}/{study_name}.pgn fi'

    if book != '':
        options += f' -openings file={book}'

    if draw is not None:
        options += f' {draw}'

    if resign is not None:
        options += f' {resign}'

    cutechess_args = f'-engine {fcp} -engine {scp} {options}'
    command = '%s %s' % (cutechess_cli_path, cutechess_args)

    logging.debug(f'match command line: {command}')

    process = Popen(command, shell = True, stdout = PIPE)
    output = process.communicate()[0]
    if process.returncode != 0:
        sys.stderr.write('failed to execute command: %s\n' % command)
        return 2

    result = ''
    for line in output.decode("utf-8").splitlines():
        logging.debug(line)
        if line.startswith('Finished match'):
            break

        if line.startswith('Score of'):
            result = line.split(': ')[1]
            result = result.split('[')[1].split(']')[0]

    sys.stdout.write(f'result {result}\n')


if __name__ == "__main__":
    main(sys.argv[1:])
