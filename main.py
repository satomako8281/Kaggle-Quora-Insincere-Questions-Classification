idx = 1
(train_idx, valid_idx) = splits[0]
with timer("fitting on {}th split".format(idx)):
    X_train = X_nb[train_idx]
    y_train = y[train_idx]
    X_val = X_nb[valid_idx]
    y_val = y[valid_idx]
    model = LogisticRegression(solver='lbfgs', dual=False, class_weight='balanced', C=0.5, max_iter=40)
    model.fit(X_train, y_train)
    models.append(model)
    valid_pred = model.predict_proba(X_val)
    train_meta[valid_idx] = valid_pred[:,1]
    test_meta += model.predict_proba(X_test_nb)[:,1] / len(splits)


df_train.pkl
df_test.pkl
pre_df_train.pkl
pre_df_test.pkl
pre2_df_train.pkl
pre2_df_test.pkl
embedding_matrix.pkl
joblib.dump(train_idx, 'train_idx.pkl', compress=3)
joblib.dump(valid_idx, 'valid_idx.pkl', compress=3)
