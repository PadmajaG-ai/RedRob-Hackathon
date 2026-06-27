import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ARCHIVE = Path('[PUB] India_runs_data_and_ai_challenge.zip')
INTERNAL_JSON = 'India_runs_data_and_ai_challenge/candidates.jsonl'
OUT_DIR = Path('eda_outputs')
OUT_DIR.mkdir(exist_ok=True)

FIELDS = [
    'candidate_id',
    'years_of_experience',
    'current_title',
    'current_company',
    'current_company_size',
    'current_industry',
    'location',
    'country',
    'headline',
    'summary',
    'open_to_work_flag',
    'preferred_work_mode',
    'willing_to_relocate',
    'profile_completeness_score',
    'recruiter_response_rate',
    'avg_response_time_hours',
    'notice_period_days',
    'search_appearance_30d',
    'saved_by_recruiters_30d',
    'interview_completion_rate',
    'offer_acceptance_rate',
    'github_activity_score',
    'expected_salary_min',
    'expected_salary_max',
    'skills',
]


def load_data():
    records = []
    with zipfile.ZipFile(ARCHIVE) as zf:
        with zf.open(INTERNAL_JSON) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                prof = row['profile']
                signals = row['redrob_signals']
                skills = []
                for s in row.get('skills', []):
                    if isinstance(s, str):
                        skills.append(s.lower())
                    elif isinstance(s, dict) and s.get('name'):
                        skills.append(s['name'].lower())
                records.append(
                    {
                        'candidate_id': row['candidate_id'],
                        'years_of_experience': prof.get('years_of_experience'),
                        'current_title': prof.get('current_title'),
                        'current_company': prof.get('current_company'),
                        'current_company_size': prof.get('current_company_size'),
                        'current_industry': prof.get('current_industry'),
                        'location': prof.get('location'),
                        'country': prof.get('country'),
                        'headline': prof.get('headline'),
                        'summary': prof.get('summary'),
                        'open_to_work_flag': signals.get('open_to_work_flag'),
                        'preferred_work_mode': signals.get('preferred_work_mode'),
                        'willing_to_relocate': signals.get('willing_to_relocate'),
                        'profile_completeness_score': signals.get('profile_completeness_score'),
                        'recruiter_response_rate': signals.get('recruiter_response_rate'),
                        'avg_response_time_hours': signals.get('avg_response_time_hours'),
                        'notice_period_days': signals.get('notice_period_days'),
                        'search_appearance_30d': signals.get('search_appearance_30d'),
                        'saved_by_recruiters_30d': signals.get('saved_by_recruiters_30d'),
                        'interview_completion_rate': signals.get('interview_completion_rate'),
                        'offer_acceptance_rate': signals.get('offer_acceptance_rate'),
                        'github_activity_score': signals.get('github_activity_score'),
                        'expected_salary_min': None if not signals.get('expected_salary_range_inr_lpa') else signals['expected_salary_range_inr_lpa'].get('min'),
                        'expected_salary_max': None if not signals.get('expected_salary_range_inr_lpa') else signals['expected_salary_range_inr_lpa'].get('max'),
                        'skills': skills,
                    }
                )
    return pd.DataFrame(records)


def save_text_summary(df):
    with open(OUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f'total candidates: {len(df)}\n')
        f.write('=== basic distributions ===\n')
        f.write(str(df[['years_of_experience', 'profile_completeness_score', 'recruiter_response_rate', 'avg_response_time_hours', 'notice_period_days', 'search_appearance_30d', 'saved_by_recruiters_30d', 'interview_completion_rate', 'offer_acceptance_rate', 'github_activity_score']].describe()))
        f.write('\n\nopen_to_work counts:\n')
        f.write(str(df['open_to_work_flag'].value_counts(dropna=False)))
        f.write('\n\npreferred_work_mode counts:\n')
        f.write(str(df['preferred_work_mode'].value_counts(dropna=False)))
        f.write('\n\nwilling_to_relocate counts:\n')
        f.write(str(df['willing_to_relocate'].value_counts(dropna=False)))
        f.write('\n\ntop industries:\n')
        f.write(str(df['current_industry'].value_counts().head(30)))
        f.write('\n\ntop titles:\n')
        f.write(str(df['current_title'].value_counts().head(30)))
        f.write('\n\ntop skills:\n')
        skill_counts = Counter([s for skills in df['skills'] for s in skills])
        for skill, count in skill_counts.most_common(40):
            f.write(f'{skill}: {count}\n')


def plot_distributions(df):
    def save_plot(fig, name):
        fig.savefig(OUT_DIR / name, bbox_inches='tight', dpi=150)
        plt.close(fig)

    numeric_cols = [
        'years_of_experience',
        'profile_completeness_score',
        'recruiter_response_rate',
        'avg_response_time_hours',
        'notice_period_days',
        'search_appearance_30d',
        'saved_by_recruiters_30d',
        'interview_completion_rate',
        'offer_acceptance_rate',
        'github_activity_score',
        'expected_salary_min',
        'expected_salary_max',
    ]
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        df[col].dropna().hist(bins=40, ax=ax)
        ax.set_title(col)
        save_plot(fig, f'{col}.png')

    for cat in ['current_industry', 'current_company_size', 'current_title', 'preferred_work_mode', 'open_to_work_flag', 'willing_to_relocate']:
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = df[cat].value_counts().head(20)
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Top {cat}')
        save_plot(fig, f'{cat}.png')

    # top skills
    skill_counts = Counter([s for skills in df['skills'] for s in skills])
    top_skills = pd.Series(dict(skill_counts.most_common(40)))
    fig, ax = plt.subplots(figsize=(10, 8))
    top_skills.plot(kind='barh', ax=ax)
    ax.invert_yaxis()
    ax.set_title('Top 40 skills')
    save_plot(fig, 'top_skills.png')


def main():
    print('loading data...')
    df = load_data()
    print('rows loaded:', len(df))
    save_text_summary(df)
    print('summary saved to', OUT_DIR / 'summary.txt')
    plot_distributions(df)
    print('plots saved to', OUT_DIR)


if __name__ == '__main__':
    main()
