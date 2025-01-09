import pathlib
from StratifiedKFoldsCV import StratifiedKFoldsCV

assignment_data_path = f"{pathlib.Path().resolve()}\\makaleler-yazarlar"
turkish_stopwords = [
    "acaba", "ama", "ancak", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey",
    "biz", "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "eğer", "en", "gibi",
    "hem", "hep", "hepsi", "her", "hiç", "ile", "ise", "kez", "ki", "kim", "mı", "mu",
    "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye", "o",
    "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani"
]

config = {
    'data_base_path': assignment_data_path,
    'K': 10,
    'stop_words': turkish_stopwords,
    'only_alpha': True,
    'split_regex': ' '
}

skfcv = StratifiedKFoldsCV(**config)

print(skfcv.class_data_count_per_fold)

sum = 0
for key in skfcv.class_data_count_per_fold:
    sum += skfcv.class_data_count_per_fold[key]
print(sum)
