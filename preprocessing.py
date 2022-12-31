import os
import re
from pprint import pprint
from typing import Dict, List, Union


def get_file_handles(training: bool = True, gold: bool = False) -> Dict[str, str]:
    """
        Retrieve file handles for training and validation reports and gold summaries
    :param training: bool
    :param gold: bool
    :return: dict of file paths
    """
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    data_type = 'gold_summaries/' if gold else 'annual_reports/'
    path += data_type

    if not os.path.exists(path):
        print(f'Specified path: {path} does not exist')
        return {}
    file_handles = {}
    for f in os.listdir(path):
        report_id = f.replace('.txt', '')
        file_handles[report_id] = f'{path}{f}'
    return file_handles


def get_gold_summaries_file_handles(file_id, training: bool = True) -> List[str]:
    """
        There are a few gold summaries per report, and they are enumerated as <report>_<#summary>.txt
    :param file_id: str/int
    :param training: bool
    :return: list of file paths
    """
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'gold_summaries/'
    if not os.path.exists(path):
        print(f'Specified path: {path} does not exist')
        return []

    file_handles = []
    for f in os.listdir(path):
        report_id = f.split('_')[0]
        if str(report_id) == str(file_id):
            file_handles.append(f'{path}{f}')
    return sorted(file_handles)


def clean_company_name(line: str):
    line = line.strip()
    # Clear of irrelevant strings
    reg_to_drop = r'''(?x) # flag to allow comments and multi-line regex
            Annual | Report | Accounts | Financial | Statements | Chairman | Executive
    '''
    pattern = re.compile(reg_to_drop, re.IGNORECASE)
    line = pattern.sub("", line)
    # Extract the name of the company
    name = line.split('plc')[0] + ' plc '
    # Try to match the year on the line and add to the identifier
    year = re.findall(r'\d{4}', line)
    if year:
        name += year[0]
    # Ensure unnecessary spaces are removed
    name = " ".join(name.split())
    return name


def get_company_from_id(file_id, training: bool = True) -> Union[str, None]:
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'annual_reports/'
    path += str(file_id)
    path += '.txt'
    max_line_iter = 100  # assume company name is before the 100th line to save on reading time
    with open(path) as file:
        for i, line in enumerate(file):
            if ' plc' in line:
                return clean_company_name(line)
            if i > max_line_iter:
                print('Name not found')
                return None


def get_id_to_company_mapping(training: bool = True) -> Dict[str, str]:
    """
        Many-to-one relationship, as reports are issued per year for a company
    :param training:
    :return:
    """
    path = 'data/'
    data_type = 'training/' if training else 'validation/'
    path += data_type
    path += 'annual_reports/'
    id_to_company_dict = {}
    for f in os.listdir(path):
        report_id = f.replace('.txt', '')
        id_to_company_dict[report_id] = get_company_from_id(report_id)
    return id_to_company_dict


def main():
    # print(get_file_handles()['17'])
    # file_handles = get_gold_summaries_file_handles(17)
    # print(file_handles)
    # print(get_company_from_id('17'))
    pprint(get_id_to_company_mapping())


main()
