

# def one_time_fix():
#     for file_path in train_files:
#         df = pd.read_csv(file_path, sep=",")
#         fn_to_numeric = partial(pd.to_numeric, errors="coerce")
#         df = df.apply(fn_to_numeric).dropna().astype(dtype=np.float32)
#         out_file = abspath(preprocessed_dir + basename(file_path))
#         if not exists(preprocessed_dir):
#             mkdir(preprocessed_dir)
#         df.to_csv(out_file, mode='w+')
#         print(f'written to {out_file}')
# one_time_fix()
