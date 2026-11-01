"""Va lib/categories.py: them danh sach lop cho cac ten bo du lieu cua de tai
({bo}_{K}shot_s{k} voi K in 1,5,10,30 va k in 0..9; {bo}_test_p3). Ban goc chi co ten chinh thuc
-> devit.py tra SEEN_CLS_DICT[DATASETS.TRAIN[0]] se loi KeyError voi lan lay mau tu sinh. Idempotent."""
import os

TEP = os.path.expanduser("~/CDFSOD-benchmark/lib/categories.py")
DAU = "# [CDFSOD-VA] ten bo du lieu cua de tai"
src = open(TEP, encoding="utf-8").read()
if DAU in src:
    print("da va tu truoc")
else:
    them = f'''
{DAU}
for dataset in datasets_name:
    for _K in (1, 5, 10, 30):
        for _k in range(10):
            SEEN_CLS_DICT[f'{{dataset}}_{{_K}}shot_s{{_k}}'] = CLASS_NAME[f'{{dataset}}_classes']
            ALL_CLS_DICT[f'{{dataset}}_{{_K}}shot_s{{_k}}'] = CLASS_NAME[f'{{dataset}}_classes']
    SEEN_CLS_DICT[f'{{dataset}}_test_p3'] = CLASS_NAME[f'{{dataset}}_classes']
    ALL_CLS_DICT[f'{{dataset}}_test_p3'] = CLASS_NAME[f'{{dataset}}_classes']
'''
    moc = "\nif __name__ == \"__main__\":"
    if moc in src:
        src = src.replace(moc, them + moc, 1)
    else:
        src = src + them
    open(TEP, "w", encoding="utf-8").write(src)
    print("da va xong")
