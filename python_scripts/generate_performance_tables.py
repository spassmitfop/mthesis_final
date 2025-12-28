# generates tables showing the agents' performances
from extract_scores_from_json import *
# In extract_scores_from_json, you can specify which jsons to use for generating the tables
from scipy import stats
import math

print_latex_tables = False
print_wilcoxon_results = False

if print_wilcoxon_results:
    for baseline, wrappers in baselines.items():
        for wrapper in wrappers:
            if baseline in relevant_wrappers and wrapper in relevant_wrappers:
                baseline_greater = 0
                wrapper_greater = 0
                list_b_greater = []
                list_w_greater = []
                undecided = 0
                for game in games:
                    if game == "disable_enemies":
                        continue
                    rewards_baseline = np.array(scores_raw_all_games[game][baseline]).flatten()

                    rewards_wrapper = np.array(scores_raw_all_games[game][wrapper]).flatten()

                    min_len = min(len(rewards_baseline), len(rewards_wrapper))

                    rewards_baseline = rewards_baseline[:min_len]
                    rewards_wrapper = rewards_wrapper[:min_len]

                    result = stats.wilcoxon(rewards_wrapper, rewards_baseline, alternative='greater')
                    print(f"{game}, H_1: {wrapper} > {baseline} :", result)
                    if not math.isnan(result[1]):
                        if result[1] < 0.05:
                            wrapper_greater += 1
                            list_w_greater.append(game)

                        result2 = stats.wilcoxon(rewards_wrapper, rewards_baseline, alternative='less')
                        if result2[1] < 0.05:
                            baseline_greater += 1
                            list_b_greater.append(game)

                        if result2[1] + result[1] != 1.0:
                            print(result2[1] + result[1])
                            print(f"{game}, H_1:  {wrapper} < {baseline} :", result2)
                        if not result[1] < 0.05 and not result2[1] < 0.05:
                            undecided += 1

                if undecided + wrapper_greater + baseline_greater != len(modifs) + len(base_games):
                    print(undecided, wrapper_greater, baseline_greater, len(base_games), len(modifs))
                    raise Exception("Something does not add up")
                print("-------------------------------------------------")
                print(f'{wrapper}: {wrapper_greater}')
                print(f'{baseline}: {baseline_greater}')
                print(f'undecided: {undecided}')
                print("Wrapper greater: ", list_w_greater)
                print("Baseline greater: ", list_b_greater)
                print("-------------------------------------------------")



if print_latex_tables:
    for baseline, wrappers in baselines.items():
        s0 = f"""
        \\begin{{table}}[H]
\centering        
\\caption{{Results of Wilcoxon signed-rank tests with \\textbf{{{wrapper_to_name_latex[baseline]}}} as the baseline {"(3 Seeds)"}}}
\\label{{tab:wsr_{baseline}_{"3"}}}
\\rowcolors{{2}}{{gray!15}}{{white}}
\\begin{{tabular}}{{l c c c}}
\\toprule
\\textbf{{Name of Wrapper}} & \\textbf{{> Baseline}} & \\textbf{{< Baseline}} & \\textbf{{Undecided}} \\\\
\\midrule
        """
        l = []

        for wrapper in wrappers:
            if baseline in relevant_wrappers and wrapper in relevant_wrappers:
                baseline_greater = 0
                wrapper_greater = 0
                list_b_greater = []
                list_w_greater = []
                undecided = 0
                for game in games:
                    if game == "disable_enemies":
                        continue
                    rewards_baseline = np.array(scores_raw_all_games[game][baseline]).flatten()

                    rewards_wrapper = np.array(scores_raw_all_games[game][wrapper]).flatten()

                    # later to be deleted
                    min_len = min(len(rewards_baseline), len(rewards_wrapper))

                    rewards_baseline = rewards_baseline[:min_len]
                    rewards_wrapper = rewards_wrapper[:min_len]

                    result = stats.wilcoxon(rewards_wrapper, rewards_baseline, alternative='greater')
                    if not math.isnan(result[1]):
                        if result[1] < 0.05:
                            wrapper_greater += 1
                            list_w_greater.append(game)
                        result2 = stats.wilcoxon(rewards_wrapper, rewards_baseline, alternative='less')
                        if result2[1] < 0.05:
                            baseline_greater += 1
                            list_b_greater.append(game)
                        if not result[1] < 0.05 and not result2[1] < 0.05:
                            undecided += 1

                max_value = max(wrapper_greater, baseline_greater, undecided)
                wrapper_greater_string = highlight_value(wrapper_greater) if wrapper_greater == max_value else str(
                    wrapper_greater)
                baseline_greater_string = highlight_value(baseline_greater) if baseline_greater == max_value else str(
                    baseline_greater)
                undecided_string = highlight_value(undecided) if undecided == max_value else str(
                    undecided)

                # Append the row to the LaTeX table
                l.append(
                    f"{wrapper_to_name_latex[wrapper]} & {wrapper_greater_string} & {baseline_greater_string} & {undecided_string} \\\\"
                )

        s2 = "\n".join(l)
        s3 = """
\\bottomrule
\\end{tabular}
\\end{table}
                    """
        print(s0 + s2 + s3)



def print_latex_table(t, text="Mean MNS over all environments."):
    s1 = """
    \\begin{table}[h!]
\\centering
\\caption{""" + text + "}"
    s1 += """
\\label{tab:mns}
\\rowcolors{2}{gray!20}{white} % <-- alternating row colors (light gray/white)
\\begin{tabular}{
    l
    S[table-format=1.3]
}
\\toprule
\\textbf{Wrapper} & \\textbf{Mean MNS} \\\\
\\midrule
        """
    l = []
    for wrapper, score in t.items():
        l.append(f"{wrapper_to_name_latex[wrapper]} & {round(score, 3):.3f} \\\\ ")
    s2 = "\n".join(l)
    s3 = """
\\end{tabular}
\\end{table}
    """
    print(s1 + s2 + s3)


def return_latex_tabular(t):
    s1 = """
\\rowcolors{2}{gray!20}{white} % <-- alternating row colors (light gray/white)
\\begin{tabular}{
    l
    S[table-format=1.3]
}
\\toprule
\\textbf{Wrapper} & \\textbf{Mean MNS} \\\\
\\midrule
        """
    l = []
    for wrapper, score in t.items():
        l.append(f"{wrapper_to_name_latex[wrapper]} & {round(score, 3):.3f} \\\\ ")
    s2 = "\n".join(l)
    s3 = """
\\end{tabular}
    """
    return s1 + s2 + s3


def combine_2_tabulars(t1, t2):
    s1 = """
\\begin{table}[h!]
\centering
\\begin{minipage}{0.46\\textwidth}
\centering """
    s1 += t1
    s1 += """
\\caption{Mean MNS over all original games}
\\label{tab:mns_base}
\\end{minipage}\\hfill
\\begin{minipage}{0.46\\textwidth}
\\centering """
    s1 += t2
    s1 += """
\\caption{Mean MNS over all modified games}
\\label{tab:mns_mods}
\\end{minipage}
\\end{table}  
"""
    print(s1)


if print_latex_tables:
    print_latex_table(normalized_scores_with_random)
    t1 = return_latex_tabular(normalized_scores_with_random_basegames)
    t2 = return_latex_tabular(normalized_scores_with_random_mods)
    combine_2_tabulars(t1, t2)


print_dict(normalized_scores_per_seed)
print("----------------------")
print_dict(normalized_scores_with_random)
print("---------------------------")

''' 
print_dict(normalized_scores_with_random_basegames)
print("--------------------------")
print_dict(normalized_scores_with_random_mods)

for g, d in scores_iqm_per_seed_all_games.items():
    for k, v in d.items():
        if k == "masked_dqn_bin_obj":
            print(f"Game {g}: {v}")

print("----------------------")
print_dict(scores_iqm_per_seed_all_games["amidar"])
'''
