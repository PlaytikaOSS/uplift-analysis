from typing import Tuple, Dict
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from uplift_analysis import scoring, evaluation, visualization


class TreatmentEffectDataGenerator:
    def __init__(self,
                 response_func: Dict,
                 num_treatments: int = 1,
                 feat_dim: int = 50,
                 hypercube_bounds: Tuple = (0, 10),
                 effect_magnitude: float = 0.3,
                 noise_var: float = 0.8):

        self.a_bounds = response_func['a']
        self.b_bounds = response_func['b']

        self.num_treatments = num_treatments
        self.feat_dim = feat_dim
        self.hypercube_bounds = hypercube_bounds
        self.effect_magnitude = effect_magnitude
        self.noise_var = noise_var

        # systematic dependence
        self.a = np.random.uniform(low=self.a_bounds[0], high=self.a_bounds[1],
                                   size=(self.feat_dim,))
        self.b = np.random.uniform(low=self.b_bounds[0], high=self.b_bounds[1],
                                   size=(self.feat_dim, self.feat_dim))
        self.c = np.random.uniform(low=self.hypercube_bounds[0],
                                   high=self.hypercube_bounds[1],
                                   size=(self.feat_dim, self.feat_dim))

    def generate(self, num_samples: int) -> Dict:

        feats = np.random.uniform(low=self.hypercube_bounds[0],
                                  high=self.hypercube_bounds[1],
                                  size=(num_samples, self.feat_dim))

        treatments_assignment = np.random.randint(self.num_treatments + 1,
                                                  size=(num_samples,))

        f = 0
        for idx in range(self.feat_dim):
            expit = -np.dot(np.abs(feats - self.c[idx, ...]), self.b[idx, ...])
            f += self.a[idx] * np.exp(expit)

        treatment_effect = 0
        for idx in range(self.num_treatments):
            treatment_effect += np.random.uniform(low=0.0, high=self.effect_magnitude * feats[..., idx]) * \
                                (treatments_assignment == idx + 1).astype(float)

        noise = np.random.normal(scale=self.noise_var, size=(num_samples,))

        response = f + noise + treatment_effect

        return {
            'features': feats,
            'treatments': treatments_assignment,
            'f': f,
            'treatment_effect': treatment_effect,
            'response': response
        }


def main():
    np.random.seed(2023)
    num_treatments = 4
    num_samples = 50000
    binary_thresh = 5.0

    dg = TreatmentEffectDataGenerator(response_func={'a': (0, 0.025), 'b': (-0.05, 0.0275)},
                                      num_treatments=num_treatments,
                                      feat_dim=50,
                                      hypercube_bounds=(0, 10),
                                      effect_magnitude=0.1,
                                      noise_var=0.8)

    num_samples = 50000
    train_set = dg.generate(num_samples)
    valid_set = dg.generate(num_samples)

    train_set['binary_response'] = (train_set['response'] > binary_thresh).astype(int)
    valid_set['binary_response'] = (valid_set['response'] > binary_thresh).astype(int)

    train_df = pd.DataFrame({k: v for k, v in train_set.items() if k != 'features'})
    valid_df = pd.DataFrame({k: v for k, v in valid_set.items() if k != 'features'})

    fig, ax = plt.subplots()
    valid_df['binary_response'].hist(ax=ax, density=True, bins=100)
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots()
    valid_df.groupby('treatments').binary_response.value_counts(normalize=True).unstack(0).plot.barh(ax=ax)
    ax.grid(True)
    plt.show()

    # T - learner
    treat_dependent_models = []
    for treat in tqdm(range(num_treatments + 1)):
        subset_idx = train_set['treatments'] == treat
        treat_dependent_models.append(
            GradientBoostingClassifier(n_estimators=10).fit(X=train_set['features'][subset_idx],
                                                            y=train_set['binary_response'][subset_idx]))

    submodels_outputs = []
    for treat in tqdm(range(num_treatments + 1)):
        submodels_outputs.append(treat_dependent_models[treat].predict_proba(valid_set['features'])[:, -1])
    valid_set['separate_learners_outptus'] = np.stack(submodels_outputs, axis=-1)

    fig, ax = plt.subplots()
    for treat in tqdm(range(num_treatments + 1)):
        ax.hist(valid_set['separate_learners_outptus'][:, treat],
                bins=200, density=True, alpha=0.3, label=treat)
    ax.grid(True)
    ax.legend()
    plt.show()

    # ======================
    # ~~~~~~~~~~~~~~~~~~~~
    # Score
    # ======================
    # ~~~~~~~~~~~~~~~~~~~~
    scorer = scoring.Scorer({'name': 't_learner',
                             'scoring_field': 'separate_learners_outptus',
                             'reference_field': 'separate_learners_outptus',
                             'reference_idx': 0,
                             'scoring_func': 'cont_score_calc'})
    ranking, recommended_action, score, action_dim = scorer.calculate_scores(dataset=valid_set)
    scored_df = pd.DataFrame(
        {
            'recommended_action': recommended_action,
            'score': score,
            'observed_action': valid_set['treatments'],
            'response': valid_set['response'],
            'binary_response': valid_set['binary_response'],
        }
    )

    # ======================
    # ~~~~~~~~~~~~~~~~~~~~
    # eval
    # ======================
    # ~~~~~~~~~~~~~~~~~~~~
    evaluator = evaluation.Evaluator(score_field='score',
                                     observed_action_field='observed_action',
                                     proposed_action_field='recommended_action',
                                     response_field='binary_response')
    scored_df, summary = evaluator.evaluate_set(scored_df=scored_df)
    another_summary = evaluator.summarize_evaluation(eval_df=scored_df)

    visualization.visualize_selection_distribution(scored_df, column_name='recommended_action')
    plt.show()

    visualization.visualize_multiple_actions_uplift_metrics(scored_df)
    plt.show()


if __name__ == '__main__':
    main()
