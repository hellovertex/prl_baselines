"""
Goal is to set up a comparison pipeline

take each individual player

for each observation, compare actions taken

/home/sascha/Documents/github.com/prl_baselines/data/baseline_model_ckpt.pt

"""
import glob
import multiprocessing
import time
from functools import partial

from prl.baselines.deprecated import hsmithy_extractor

best_players_over20k_gains = ['ishuha',
                              'Sakhacop',
                              'nastja336',
                              'Lucastitos',
                              'I LOVE RUS34',
                              'SerAlGog',
                              'Ma1n1',
                              'zMukeha',
                              'SoLongRain',
                              'LuckyJO777',
                              'Nepkin1',
                              'blistein',
                              'ArcticBearDK',
                              'Creator_haze',
                              'ilaviiitech',
                              'm0bba',
                              'KDV707']
best_players_over5k_gains = ['Maximysss85', 'Fumizja', 'Rusyaha', 'KimWanHull', 'kolykam', 'Fabian_PL', 'Reg1N55',
                             'mantas79', 'riddle017', 'vadjkeee', 'Punzzzer', 'Shaftey11', 'symkalaila', 'OJIEHb222',
                             'Jzzz99', 'RichRiu4', 'JahSoldier69', 'illarins', 'MurdocPwR', 'realphenom', 'MaCe90',
                             'danuuutz', 'philhell24', 'delpalo', 'ChrisChamber', 'bulhakau', 'fire9372', 'Justice2k17',
                             'R_Silis93', 'extinction51', 'sven219', 'Myk2004', 'TheFranchiss', 'shootingGRID',
                             'Geconiel', 'VirteKs', 'bigBUG1488', 'Bratishok', 'chechoraiser', 'yzuk', 'epiLewg',
                             'master_pc06', 'onepiece314', 'Naughty_ept', 'Fosca999', 'badmadrabbit', '4ikibumbella',
                             'LilTonyV', 'easy_sarcasm', 'clww', 'X-Triha-X', 'spartaknv', 'Voodgchoo', 'andutu34',
                             'IC1MALE', 'lol19899', 'Rysiff', 'Heaton#RISES', 'morraish', 'fate_eyed', 'bekonja',
                             'vIpClubFish', 'VR13', 'GeoflexPoker', 'Red.Jh0nn', 'Pegasus2102', 'Korifei94',
                             'rusher_NiNja', 'Foccc', 'XxDragomanxX', 'trewq123', 'nepkin', 'doselka', 'Iuckyfluke',
                             'dexeosRankX', 'WYIYW', 'ShoutToDeath', 'shnur221087', 'jardinero9', '2pokTG', 'spensor1',
                             'dawnerr', 'Retschy', 'vadimstryha', 'sisojla', 'DirtyRuble', 'danycreep', 'Communist654',
                             'mind.in.aBox', 'LichtVS', 'GaussJordan24', 'xxxZeka', 'odisseas1989', 'SugarRayLeon',
                             'N_Rimer', 'ivan_graf', 'VCardKiller', 'FuchsJ', 'MoZL9', 'SGralex', 'Lakaem45', 'Y_19',
                             'fish my nuts', 'max21rus1988', 'MARGARCIA09', 'Tigristen', 'Julnin', 'reeqar',
                             'Kach_2010', 'voshod561', 'Hidey7', 'geopaok4', '23RoChe23', 'Aleqseuka', 'scenotaph',
                             'Icefire00627', 'fandorin2005', 'jwu63', 'irenicus24', 'Winterds', 'xSilverKx',
                             'FreshLocky', 'santa88188', 'Shirachi90', 'manashov', 'ACnuJkee', 'maxxll', 'BorissKGB',
                             'Chess_pub', 'kaiaman', 'Lange1es', 'sircatres', 'Kolyan2114', 'luis619angel',
                             'Gallofree333', 'iai86', 'MaximUSNG', 'Sanctigallus', 'mradon', 'Mr. Itsco', 'Jo_jpp',
                             'RCAMDESSUS', 'KOCTA47', 'Fantom979', 'pomkaPWNZ', 'worth303', 'Dspn', 'solbi86', 'Pewbie',
                             'shameShamee', 'Bkmbx86', 'WINDTOY', 'santamariya89', 'spr1teg', 'HELOTS', 'prophet436',
                             'Becar_06', 'One23drumm', 'jjohnys85', 'Soulfrozz', 'zver1596357', 'PinnacleUa', 'pupan45',
                             'salamandryko', 'fcardu25', 'N1leon', 'Russia163', 'Furbzzzzz', 'BefirsSt', 'DUICA611',
                             'crist_89870', 'grahpAA', 'derek5858', 'ChikoKz740', 'ExperimentoPS', 'soplak',
                             'RiggedBoy', 'feelgroundik', 'yorke132', 'aandrushaa', 'edmundtrebus', 'Radzko', 'ruQx01',
                             'borneodreams', 'jext', 'drosser321', 'Thxmoon1', 'olo0olo', 'razukhin', 'malabar357',
                             'kvitun', 'GabOo175', 'BenjiPL', 'Roma_Demiden', 'lebowskiguy', 'Bostya', 'Anton Tr',
                             'drakula110', 'Lurst', 'gitarist74', 'orobertino', 'Gerafont', 'Dioprest', 'koksskrt',
                             '1Spanec', 'quadroq3', 'Gluck35', 'nguyenthianhdao', 'Daorin', 'Becks Baker',
                             'Maks Bavenko', 'Eretik1985', 'MaksCh1', 'tommyteckers', 'goldmel', 'eeeLOVEnkov',
                             'eyesellbeams', 'Marklar1992', 'BinaryStorm', 'Pgaliley', 'AceKingMir', 'Faine mist0',
                             'O_Buda', 'Gaben ICM', 'tyabutani', '1TOP1', 'shaykee', 'DXJ7', 'Subaru52rus',
                             'dontbetme27', 'Degtyaryov', 'Yaruk7', 'PuRpL_M4mbA', 'Rezzonner', 'vest78', 'Smogg96',
                             'alf51321', 'LJGDMBC', '7bekon7', 'Heidy_T', 'super_pavel9', 'eraser2308', 'patriot424',
                             'chikan0k', 'G.Andrei95', 'PotatoMonste', 'MyBestPokeR', 'Belvedera', 'Infusorium',
                             'sil3ntium8', 'Strelok1974', 'mixailll99', 'Shivas prana', 'Yspenyanin', 'Solovyova',
                             'vampire131313', 'MordOleg', 'EasyGamerrr', 'Kuemmerlinge', 'Apuctokpat777', 'wanax777',
                             'Maxi Goldman', 'Xenicide', '1kum', 'juipelo', 'Ext.HuliGan', '0neLevel', 'dragu419',
                             'SATAN0SS', 'EddyFelson90', 'Andrew Dee F', 'jimbo_fast', 'Redman3008', 'aldair68',
                             'vbnote', 'oxxxy', 'AndreiCoz', 'duk555', 'rgkkk', 'irc122', 'MarderIII', 'L.A.Casillaz',
                             'Arlongur', 'CashfishT', 'dimakryt1', 'Ikmpnp', 'qpuwapa', '03upuc', 'uuzka',
                             'GranMaster87', 'censored2013', 'ibaCker', 'cool9393', 'Bu11iT', 'orbcrazyk', 'BorisRoyaL',
                             'andru_rock', 'Lin_Habey', 'ooak182', 'bandluchok', 'vestimokrec', 'Stepaniuk',
                             'Pavel Podkur', 'S1HED', 'skars07', 'AstrumNatale', '13tochka5', 'pabliq9', 'G.HANSEN333',
                             'pashka148', 'FRED47056', 'Ladybird1367', 'Sensei2212', 'BELGAZ', 'kozirek', 'Godlike1379',
                             'Geka4an', 'titanik80', 'Kreayshawn', 'Lucastitos', 'Creator_haze', 'blistein', 'zMukeha',
                             'I LOVE RUS34', 'Nepkin1', 'SoLongRain', 'ilaviiitech', 'KDV707', 'ArcticBearDK', 'Ma1n1',
                             'SerAlGog', 'm0bba', 'LuckyJO777', 'nastja336', 'ishuha', 'Sakhacop']
gains = {"ishuha": {"n_hands_played": 59961, "n_showdowns": 18050, "n_won": 10598, "total_earnings": 37425.32000000013},
         "Sakhacop": {"n_hands_played": 54873, "n_showdowns": 14718, "n_won": 9235,
                      "total_earnings": 43113.86000000007},
         "nastja336": {"n_hands_played": 48709, "n_showdowns": 11231, "n_won": 7303,
                       "total_earnings": 35729.6900000002},
         "Lucastitos": {"n_hands_played": 37898, "n_showdowns": 11117, "n_won": 6811,
                        "total_earnings": 20171.329999999984},
         "I LOVE RUS34": {"n_hands_played": 36457, "n_showdowns": 10985, "n_won": 6441,
                          "total_earnings": 21504.00999999993},
         "SerAlGog": {"n_hands_played": 50103, "n_showdowns": 10850, "n_won": 6613,
                      "total_earnings": 25631.720000000074},
         "Ma1n1": {"n_hands_played": 40296, "n_showdowns": 9792, "n_won": 6188, "total_earnings": 25016.60999999976},
         "zMukeha": {"n_hands_played": 34991, "n_showdowns": 9104, "n_won": 6003, "total_earnings": 21469.710000000083},
         "SoLongRain": {"n_hands_played": 33826, "n_showdowns": 8722, "n_won": 5381,
                        "total_earnings": 22247.390000000087},
         "LuckyJO777": {"n_hands_played": 33201, "n_showdowns": 8118, "n_won": 5283,
                        "total_earnings": 26579.860000000004},
         "Nepkin1": {"n_hands_played": 28467, "n_showdowns": 8032, "n_won": 5501, "total_earnings": 21739.989999999976},
         "blistein": {"n_hands_played": 34620, "n_showdowns": 7966, "n_won": 5134, "total_earnings": 20824.44000000001},
         "ArcticBearDK": {"n_hands_played": 24449, "n_showdowns": 6849, "n_won": 4292,
                          "total_earnings": 24626.509999999973},
         "Creator_haze": {"n_hands_played": 23882, "n_showdowns": 6737, "n_won": 4172,
                          "total_earnings": 20679.31000000002},
         "ilaviiitech": {"n_hands_played": 29527, "n_showdowns": 6401, "n_won": 4213,
                         "total_earnings": 22407.82999999994},
         "m0bba": {"n_hands_played": 24384, "n_showdowns": 6349, "n_won": 4325, "total_earnings": 25772.450000000015},
         "KDV707": {"n_hands_played": 25570, "n_showdowns": 6241, "n_won": 3492, "total_earnings": 23929.970000000063}}

folder_out = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_336"
extr = hsmithy_extractor.HSmithyExtractor()
filenames = glob.glob("/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped/**/*.txt",
                      recursive=True)
# n_files = len(filenames)
# n_files_skipped = 0
# for i, f in enumerate(filenames):
#     print(f'Extractin file {i} / {n_files}')
#     for pname in best_players_over5k_gains:
#         try:
#             extr.extract_file(file_path_in=f,
#                               file_path_out=folder_out,
#                               target_player=pname)
#         except UnicodeDecodeError:
#             n_files_skipped += 1
# print(f"Done. Extracted {n_files - n_files_skipped}. {n_files_skipped} files were skipped.")
x = 10000
chunks = []
current_chunk = []
i = 0
for file in filenames:
    current_chunk.append(file)
    if (i + 1) % x == 0:
        chunks.append(current_chunk)
        current_chunk = []
    i += 1
# trick to avoid multiprocessing writes to same file
for i, chunk in enumerate(chunks):
    chunk.append(f'CHUNK_INDEX_{i}')

fn = partial(extr.extract_files, file_path_out=folder_out, target_players=best_players_over5k_gains)
start = time.time()
p = multiprocessing.Pool()
# run f0
for x in p.imap_unordered(fn, chunks):
    print(x + f'. Took {time.time() - start} seconds')
print(f'Finished job after {time.time() - start} seconds.')

p.close()

n_files = len(filenames)
n_files_skipped = 0

print(f"Done. Extracted {n_files - n_files_skipped}. {n_files_skipped} files were skipped.")