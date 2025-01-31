import pandas as pd

# Funcție pentru procesarea fișierului și generarea fișierului de rezultate
def proceseaza_fisier(fisier_pred, id_list, fisier_output):
    # Dicționar pentru a stoca sumele
    suma_dict = {tip_id: {} for tip_id in ['IN', 'PR', 'RT']}

    # Citim fișierul în bucăți pentru a evita consumul mare de memorie
    for chunk in pd.read_csv(fisier_pred, chunksize=500000):
        # Asigurăm că valorile din coloana 9 sunt numerice (0 sau 1)
        chunk.iloc[:, 8] = pd.to_numeric(chunk.iloc[:, 8].replace({0: 0, 1: 1}), errors='coerce')

        # Iterăm prin fiecare ID din lista dată
        for id_val in id_list:
            # Filtrăm rândurile care au același ID
            rows_with_id = chunk[chunk.iloc[:, 0] == id_val]

            # Verificăm dacă există rânduri cu ID-ul respectiv
            if rows_with_id.empty:
                print(f"Nicio linie găsită pentru ID: {id_val}")
                continue

            # Iterăm prin tipurile de ID-uri (IN, PR, RT)
            for tip_id in ['IN', 'PR', 'RT']:
                # Filtrăm rândurile pentru tipul de ID
                rows_with_tip = rows_with_id[rows_with_id.iloc[:, 2].str.strip().str.upper() == tip_id]

                # Verificăm dacă există rânduri cu tipul de ID
                if rows_with_tip.empty:
                    print(f"Nicio linie găsită pentru ID: {id_val}, Tip ID: {tip_id}")
                    continue

                # Calculăm suma valorilor
                suma = rows_with_tip.iloc[:, 8].sum()

                # Actualizăm suma în dicționar
                if id_val in suma_dict[tip_id]:
                    suma_dict[tip_id][id_val] += suma
                else:
                    suma_dict[tip_id][id_val] = suma

    # Creăm DataFrame-uri pentru rezultate
    id_sume_df = []
    for tip_id in suma_dict:
        for id_val, suma in suma_dict[tip_id].items():
            id_sume_df.append({'ID': id_val, 'Tip ID': tip_id, 'Sumă': suma})

    id_sume_df = pd.DataFrame(id_sume_df)

    # Determinăm suma maximă
    suma_maxima = id_sume_df['Sumă'].max()

    # Calculăm 50% din suma maximă
    jumatate_suma_maxima = suma_maxima * 0.5

    # Filtrăm ID-urile care au suma mai mare sau egală cu 50% din suma maximă
    sume_top_df = id_sume_df[id_sume_df['Sumă'] >= jumatate_suma_maxima]

    # Generăm un fișier Excel cu rezultatele
    with pd.ExcelWriter(fisier_output) as writer:
        # Scriem rezultatele pentru fiecare tip de ID într-o foaie separată
        for tip_id in ['IN', 'PR', 'RT']:
            filtered_df = sume_top_df[sume_top_df['Tip ID'] == tip_id]
            filtered_df.to_excel(writer, sheet_name=tip_id, index=False)

            # Afișăm primele 3 sume maxime și ID-urile asociate în consolă
            print(f"Tip ID: {tip_id}")
            print(filtered_df[['ID', 'Sumă']].head(3))  # Afișăm primele 3
            print("\n")

    # Confirmăm salvarea fișierului
    print(f'Rezultatele pentru sumele mai mari sau egale cu 50% din suma maximă au fost salvate în {fisier_output}')

# Citirea fișierului de ID-uri din DrugCentral
drugcentral_df = pd.read_excel('DrugCentral_ID.xlsx')

# Extragem lista de ID-uri din DrugCentral
id_list = drugcentral_df.iloc[:, 0]

# Listele fișierelor de predicții și fișierele de ieșire
fisere_pred = ['cnn-pred.csv', 'rfc-pred.csv', 'mlp-pred.csv', 'svc-pred.csv']
fisiere_output = ['CNN_rezultate_top_50suma.xlsx', 'RFC_rezultate_top_50suma.xlsx', 'MLP_rezultate_top_50suma.xlsx', 'SVC_rezultate_top_50suma.xlsx']

# Iterăm prin fiecare fișier de predicție și procesăm rezultatele
for fisier_pred, fisier_output in zip(fisere_pred, fisiere_output):
    proceseaza_fisier(fisier_pred, id_list, fisier_output)

