import json

import numpy as np
import pandas
import pandas as pd
from typing import List
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore
from pathlib import Path

PLOT_PATH: Path = Path.cwd() / 'plots'
DATA_PATH: Path = Path.cwd() / 'data'
PLOT_PATH.mkdir(exist_ok=True, parents=True)
DATA_PATH.mkdir(exist_ok=True, parents=True)


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
    fig.savefig(PLOT_PATH / f'{f_name}.png')


def percentage(part: float, whole: float) -> float:
    """
    Helper function to calculate percentages
    """
    return part * 100 / whole


def norm_accuracy_by_data_freq(acc: float, data_frequencies: List[float]) -> float:
    """
    Computes normalized accuracy scores for a given accuracy and frequency of options in the reference set
    :param acc: accuracy of annotator
    :param data_frequencies: frequency of annotation-option
    :return: normalized accuracy based on data frequency
    """
    data_frequency_factor: float = sum([frequency ** 2 for frequency in data_frequencies])
    return (acc - data_frequency_factor) / (1 - data_frequency_factor)


with open(DATA_PATH / 'anonymized_project.json', 'r') as f:
    data: dict = json.load(f)
result_set: dict = data['results']['root_node']['results']

task_results: dict = {}

# keys for dictionary access; tuple-elements at index > 0 are keys for nested dicts
column_names: List = [
    ('task_input', 'image_url'),
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
annotator_df['task_input_image_url'] = [
    (img_url.rsplit('/')[-1]).split('.')[0] for img_url in annotator_df['task_input_image_url']
]
annotator_df.set_index('user_id', inplace=True)
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

# plot annotator results
fig = plt.figure(figsize=(8, 6), dpi=300)
plt.bar(task_results['annotator_result_count'].keys(), task_results['annotator_result_count'].values(), color='#a1ccf4')
current_axes = plt.gca()
current_axes.axes.get_xaxis().set_major_locator(ticker.MultipleLocator(1))  # show every annotator on the x-axis
plt.title('Annotations by annotator')
plt.xlabel('Annotator')
plt.ylabel('Number of annotations')
fig.savefig(PLOT_PATH / 'annotator_results.png')

answer_counts_by_question: pandas.Series = annotator_df.groupby('question')['task_output_answer'].value_counts()

controversial_questions: List = []
for index, count in answer_counts_by_question.items():
    # do annotators vote 'yes' nearly as often as 'no' - adjust this range to define the term "highly disagree"
    if count in range(4, 6):
        if index[0] not in controversial_questions:
            controversial_questions.append(index[0])

task_results['controversial_questions'] = controversial_questions

# task 2
# number of responses of 'cant_solve' and 'corrupt_data' respectively
task_results['cant_solve_count'] = len(annotator_df[(annotator_df['task_output_cant_solve'] == 1)])
task_results['corrupt_data_count'] = len(annotator_df[(annotator_df['task_output_corrupt_data'] == 1)])

# dataframe for trend_analysis
unsolvable_and_corrupted_df: pd.DataFrame = annotator_df[
    (annotator_df['task_output_corrupt_data'] == 1) | (annotator_df['task_output_cant_solve'] == 1)
]


# task 3
with open(DATA_PATH / 'references.json', 'r') as f:
    references_set = json.load(f)

# new data frame for validation reference
references_df: pd.DataFrame = pd.DataFrame(
    [[img, validation['is_bicycle']] for img, validation in references_set.items()],
    columns=['task_input_image_url', 'validation'],
)

images_count: int = len(references_df)  # row_count
images_with_bicycles_count: int = references_df['validation'].value_counts()[
    True
]  # images on which a bicycle can be seen
images_without_bicycles_count: int = images_count - images_with_bicycles_count  # images on which no bicycle can be seen

task_results['reference_set'] = {}
task_results['reference_set']['images_with_bicycle'] = images_with_bicycles_count
task_results['reference_set']['images_without_bicycle'] = images_without_bicycles_count

# plot balance of reference set
labels = [f'is bicycle ({images_with_bicycles_count})', f'is not bicycle ({images_without_bicycles_count})']
sizes = [percentage(images_with_bicycles_count, images_count), percentage(images_without_bicycles_count, images_count)]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, explode=(0.05, 0), shadow=True, autopct='%1.2f%%', startangle=90)
plt.title('Ground truth distribution')
fig.savefig(PLOT_PATH / 'references_distribution.png')


# task 4
annotator_df['index_copy'] = annotator_df.index  # copy index to reindex data frame after merge
annotator_df = pd.merge(annotator_df, references_df, on='task_input_image_url')  # merge data frames on image-strings
annotator_df.set_index('index_copy', inplace=True)  # recreate original index
annotator_df.sort_index(inplace=True)  # sort numerically by annotator

annotator_stats: dict = {annotator: {} for annotator in set(list(annotator_df.index))}  # one dictionary per annotator

for annotator in set(list(annotator_df.index)):
    # get columns validation and task_output_answer for the current annotator to evaluate their accuracy
    cur_annotator_df: pd.DataFrame = annotator_df.loc[annotator][['validation', 'task_output_answer']]
    true_positive: int = len(
        cur_annotator_df[((cur_annotator_df['task_output_answer'] == 'yes') & (cur_annotator_df['validation'] == 1))]
    )
    true_negative: int = len(
        cur_annotator_df[((cur_annotator_df['task_output_answer'] == 'no') & (cur_annotator_df['validation'] == 0))]
    )
    false_positive: int = len(
        cur_annotator_df[((cur_annotator_df['task_output_answer'] == 'yes') & (cur_annotator_df['validation'] == 0))]
    )
    false_negative: int = len(
        cur_annotator_df[((cur_annotator_df['task_output_answer'] == 'no') & (cur_annotator_df['validation'] == 1))]
    )
    correct_predictions: int = true_positive + true_negative
    wrong_predictions: int = false_positive + false_negative

    annotator_stats[annotator]['correct_predictions'] = correct_predictions
    annotator_stats[annotator]['wrong_predictions'] = wrong_predictions
    annotator_stats[annotator]['accuracy'] = correct_predictions / (correct_predictions + wrong_predictions)
    annotator_stats[annotator]['precision'] = true_positive / (true_positive + false_positive)
    annotator_stats[annotator]['recall'] = true_positive / (true_positive + false_negative)

# prepare data to plot
labels = list(set(list(annotator_df.index)))
accuracies = [annotator_stats[annotator]['accuracy'] for annotator in labels]
precisions = [annotator_stats[annotator]['precision'] for annotator in labels]
recalls = [annotator_stats[annotator]['recall'] for annotator in labels]

task_results['accuracies'] = accuracies
task_results['precisions'] = precisions
task_results['recalls'] = recalls

# expected accuracy through random chance based on data frequency
expected_accuracies = [
    norm_accuracy_by_data_freq(
        acc, [images_with_bicycles_count / images_count, images_without_bicycles_count / images_count]
    )
    for acc in accuracies
]

good_annotators: List[int] = []
bad_annotators: List[int] = []

for idx, acc in enumerate(accuracies):
    if acc > np.mean(accuracies):  # good annotators
        good_annotators.append(idx + 1)
    elif acc < np.mean(accuracies) - np.std(accuracies):  # bad annotators
        bad_annotators.append(idx + 1)

fig = plt.figure(figsize=(8, 6), dpi=300)
plt.bar(labels, task_results['annotator_result_count'].values(), label='#Answered questions', color='#a1ccf4')
plt.xlabel('Annotator')
plt.ylabel('Answered Questions')
plt.title('Annotator Evaluation')
current_axes = plt.gca()
current_axes.axes.get_xaxis().set_major_locator(ticker.MultipleLocator(1))  # show every annotator on the x-axis
for idx, tick_label in enumerate(current_axes.get_xaxis().get_ticklabels()):  # color tick labels for special annotators
    if idx - 1 in bad_annotators:
        tick_label.set_color('red')
    elif idx - 1 in good_annotators:
        tick_label.set_color('green')
ax2 = plt.twinx()
ax2.plot(labels, accuracies, label='Accuracy')
ax2.plot(labels, expected_accuracies, label='Normalized accuracy')
ax2.set_ylabel('Percentage')
ax2.legend(bbox_to_anchor=(1, 1.1), loc=1, borderaxespad=0)
fig.savefig(PLOT_PATH / 'annotator_evaluation.png')
