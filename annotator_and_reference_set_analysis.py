import json
import pandas
import pandas as pd
from typing import List
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore
from pathlib import Path


def plot_annotation_times(
    df: pd.DataFrame, title: str = 'Annotation duration times', f_name: str = 'annotation_duration'
) -> None:
    """
    Plots a figure with a box plot and scatter plot for a column 'task_output_duration_ms'
    :param df: dataframe
    :param title: title for plot
    :param f_name: file_name to use
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    plt.subplots_adjust(wspace=0.3)
    fig.suptitle(f'{title}')
    ax1.boxplot(df['task_output_duration_ms'], showfliers=False, notch=True)
    ax1.get_xaxis().set_visible(False)
    ax1.set_title('Box plot for all annotator times')
    ax1.set_ylabel('Time in ms')
    # use index values for x-axis (annotators)
    ax2.scatter([index for index in list(df.index.get_level_values(0))], df['task_output_duration_ms'], s=1)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))  # show every annotator on the x-axis
    ax2.set_title('Annotation duration by annotator')
    ax2.set_xlabel('Annotator')
    ax2.set_ylabel('Time in ms')
    ax2.tick_params(axis='x', labelsize=8)
    (Path.cwd() / 'plots').mkdir(parents=True, exist_ok=True)
    fig.savefig(Path.cwd() / f'plots/{f_name}.png')


with open("data/anonymized_project.json", "r+") as f:
    data: dict = json.load(f)
result_set: dict = data['results']['root_node']['results']

task_results: dict = {}

# keys for dictionary access; tuple-elements at index > 0 are keys for nested dicts
column_names: List = [
    ('created_at',),
    ('workpackage_total_size',),
    ('loss',),
    ('task_output', 'answer'),
    (
        'task_output',
        'cant_solve',
    ),
    (
        'task_output',
        'corrupt_data',
    ),
    (
        'task_output',
        'duration_ms',
    ),
    (
        'user',
        'vendor_id',
    ),
    (
        'user',
        'vendor_user_id',
    ),
]

selected_data: List = []  # nested list for dataframe
for question, task_stats in result_set.items():  # iterate task results
    for annotation in task_stats['results']:  # iterate list of annotators for this task

        annotation_stats: List = []  # row information
        for c_name in column_names:
            if len(c_name) > 1:  # nested dict
                annotation_stats.extend([annotation[c_name[0]][c_name[1]]])
            else:
                annotation_stats.extend([annotation[c_name[0]]])
        annotation_stats.append(question)  # add the pseudomized question as last element
        selected_data.append(annotation_stats)

column_names.append(('question',))  # also add a last element for column names
# create reasonable headers for a data frame
col_names: List = [c_name[0] if len(c_name) == 1 else f"{c_name[0]}_{c_name[1]}" for c_name in column_names]
annotator_df: pd.DataFrame = pd.DataFrame(selected_data, columns=col_names)
# cast the numeric part of the annotators to integers, since an integer index is easier to work with
annotator_df['user_id'] = [int(user.rsplit('_')[-1]) for user in annotator_df['user_vendor_user_id']]
annotator_df = annotator_df.set_index('user_id')
annotator_df.sort_index(inplace=True)

annotator_count: int = len(set(list(annotator_df.index.get_level_values(0))))  # number of distinct annotators

task_results['max_annotation_duration'] = annotator_df['task_output_duration_ms'].max()  # maximum annotation duration
task_results['min_annotation_duration'] = annotator_df['task_output_duration_ms'].min()  # minimum annotation duration
task_results['mean_annotation_duration'] = annotator_df['task_output_duration_ms'].mean()  # average annotation duration
plot_annotation_times(annotator_df)

curated_df: pd.DataFrame = annotator_df[(annotator_df['task_output_duration_ms'] > 0)]  # remove negative outlier
plot_annotation_times(
    curated_df, title='Annotation duration times without negatives', f_name='annotation_duration_without_negatives'
)

curated_df = annotator_df[(annotator_df['task_output_duration_ms'] > 250)]  # adjust acceptable time to answer
plot_annotation_times(
    curated_df,
    title='Annotation duration times below average reaction time',
    f_name='annotation_duration_below_average_reaction_time',
)

# respective number of results produced by individual annotators
task_results['annotator_result_count'] = {
    index: len(annotator_df.loc[index]) for index in set(list(annotator_df.index.get_level_values(0)))
}


fig = plt.figure(figsize=(6, 3), dpi=150)
plt.bar(task_results['annotator_result_count'].keys(), task_results['annotator_result_count'].values(), color='#a1ccf4')
plt.title('Annotations by annotator')
plt.xlabel('Annotator')
plt.ylabel('Number of annotations')
fig.savefig(Path.cwd() / 'plots/annotator_results.png')

answer_counts_by_question: pandas.Series = annotator_df.groupby('question')['task_output_answer'].value_counts()

controversial_questions: List = []
for index, count in answer_counts_by_question.items():
    # do annotators vote 'yes' nearly as often as 'no' - adjust this range to define the term "highly disagree"
    if count in range(4, 6):
        if index[0] not in controversial_questions:
            controversial_questions.append(index[0])

task_results['controversial_questions'] = controversial_questions

# number of responses of 'cant_solve' and 'corrupt_data' respectively
task_results['cant_solve_count'] = len(annotator_df[(annotator_df['task_output_cant_solve'] == 1)])
task_results['corrupt_data_count'] = len(annotator_df[(annotator_df['task_output_corrupt_data'] == 1)])

# dataframe for trend_analysis
unsolvable_and_corrupted_df: pd.DataFrame = annotator_df[
    (annotator_df['task_output_corrupt_data'] == 1) | (annotator_df['task_output_cant_solve'] == 1)
]
