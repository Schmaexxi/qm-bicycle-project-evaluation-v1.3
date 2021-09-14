import unittest
from pathlib import Path
import json
import pandas as pd


class TestFileStructure(unittest.TestCase):
    def setUp(self) -> None:
        """
        Loads a json-file and assigns a nested dictionary accessed by the keys "['results']['root_node']['results']"
        """
        path_to_data: Path = Path.cwd().parent / 'data/anonymized_project.json'
        with open(path_to_data, 'r') as file:
            data: dict = json.load(file)
        self.result_set: dict = data['results']['root_node']['results']

    def test_consistent_hierarchy(self) -> None:
        """
        Checks a dictionary's structure for coherence,
        i.e., the existence of the same keys
        """
        # get arbitrary first task from dict
        arbitrary_first_task: str = list(self.result_set.keys())[0]
        arbitrary_first_result_set: dict = self.result_set.pop(arbitrary_first_task)
        arbitrary_first_result: dict = arbitrary_first_result_set['results'][0]

        # define dictionary to compare keys of an item with
        # for nested dicts, their original key holds a list of its nested dictionary's keys
        fields: dict = {
            key: [nested_key for nested_key in val.keys()]
            for key, val in arbitrary_first_result.items()
            if isinstance(val, dict)
        }

        fields['base'] = [key for key in arbitrary_first_result.keys()]  # first-level keys are accessed by "base"

        # iterate over the input data to assert the existence of the same keys
        for _, task_list in self.result_set.items():
            for task in task_list['results']:
                for key, val in task.items():
                    # check the keys in every task
                    self.assertEqual([key for key in task.keys()], fields['base'], "Keys do not match")
                    if isinstance(val, dict):  # for keys of nested dicts
                        self.assertEqual(
                            [key for key in val.keys()], fields[key], "Keys of nested json field do not match"
                        )

    def test_gui_type(self) -> None:
        """
        Checks a dictionary for mismatching gui_types
        """
        # get arbitrary first gui type from dict
        arbitrary_first_gui_type: str = self.result_set[list(self.result_set.keys())[0]]['gui_type']

        for _, results in self.result_set.items():
            self.assertEqual(
                results['gui_type'],
                arbitrary_first_gui_type,
                f"{results['gui_type']} != {arbitrary_first_gui_type}",
            )

    def test_id_user_relation(self) -> None:
        """
        Checks for a dictionary whether the keys 'vendor_id' and 'vendor_user_id' uniquely identify the key 'id'
        Assumes 'test_consistent_hierarchy' to succeed
        """
        users: dict = {
            'vendor_id': [],
            'id': [],
            'vendor_user_id': [],
        }
        # populate the defined dict with users from the results
        for question_result in self.result_set.values():
            for result in question_result['results']:
                for key, val in result['user'].items():
                    users[key].append(val)
        users_and_ids_df: pd.DataFrame = pd.DataFrame.from_dict(users)
        individual_users: set = set(list(users_and_ids_df['vendor_user_id']))  # distinct users
        # combination of 'id' and 'vendor_user_id'
        users_and_ids_count_df: pd.DataFrame = users_and_ids_df.groupby(['id', 'vendor_user_id']).count()
        # is the combination between 'id' and 'vendor_user_id' unique?
        self.assertTrue(
            len(users_and_ids_count_df) == len(individual_users),
            "Non-unique mapping exists between 'id' and 'vendor_user_id'",
        )


if __name__ == '__main__':
    unittest.main()
