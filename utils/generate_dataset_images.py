import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm

ABSOLUTE_PATH = '/home/reshyurem/Accurate-smiles-recognition-Img2Mol/'

df = pd.read_csv(ABSOLUTE_PATH + 'data/smiles.csv', delimiter='\t')

ms = []
for i in tqdm(range(len(df))):
    smi = df.iloc[i]['Smiles']
    ms.append(Chem.MolFromSmiles(smi))

for i in tqdm(range(len(ms))):
    Draw.MolToFile(ms[i], ABSOLUTE_PATH + 'data/train_images/{}.png'.format(i))

print('{} images generated'.format(len(ms)))