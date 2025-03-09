from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def extract_unique_elements(df, column_name):
    unique_elements = []
    for value in df[column_name]:
        if value is not None:
            for element in value:
                if element not in unique_elements:
                    unique_elements.append(element)
    return unique_elements


def encode_elements(df, column_name, unique_elements):
    for element in unique_elements:
        values = []
        for value in df[column_name]:
            if value is not None and element in value:
                values.append(1)
            else:
                values.append(0)
        df[element] = values
    return df


def preprocess_data(df):
    tags = extract_unique_elements(df, 'tags')
    df = encode_elements(df, 'tags', tags)

    df['ingredients_types_list'] = df['ingredients'].apply(lambda x: [ingredient['type'] for ingredient in x])
    df['ingredients_types_list'] = df['ingredients_types_list'].apply(
        lambda x: [ingredient for ingredient in x if ingredient is not None])

    types = extract_unique_elements(df, 'ingredients_types_list')
    df = encode_elements(df, 'ingredients_types_list', types)

    df = df.drop(
        ['id', 'tags', 'name', 'imageUrl', 'instructions', 'alcoholic', 'createdAt', 'updatedAt', 'ingredients',
         'ingredients_types_list'], axis=1)

    for col in df.select_dtypes(include='number').columns.tolist():
        if df[col].sum() < 12:
            df.drop(col, axis=1, inplace=True)

    preprocessing = ColumnTransformer([
        ('encoding', OneHotEncoder(), ['category', 'glass'])
    ], remainder='passthrough')

    X = preprocessing.fit_transform(df)

    return X
