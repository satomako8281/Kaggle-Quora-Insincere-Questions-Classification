from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, make_union

from transformers import (
    PandasSelector, LowerCase, TextCleaner, NumberCleaner, FillEmpty,
    QuoraTokenizer, FeaturesCapsVSLength, FeaturesWordsVSUnique, ReportShape
)


def prepare_vecorizer_1():
    vectorizer = make_pipeline(
        LowerCase(),
        TextCleaner(),
        NumberCleaner(),
        FillEmpty(),

        make_union(
            make_pipeline(
                PandasSelector(columns=['question_text']),
                QuoraTokenizer(),
            ),
            make_pipeline(
                PandasSelector(columns=['question_text']),
                FeaturesCapsVSLength(),
                StandardScaler(),
            ),
            make_pipeline(
                PandasSelector(columns=['question_text']),
                FeaturesWordsVSUnique(),
                StandardScaler(),
            )
        ),
        ReportShape()
    )

    return vectorizer

