
import pandas as pd

def process_file(input_file, id_list, output_file):
    sum_dict = {descriptor: {} for descriptor in ["IN", "PR", "RT"]}

    for chunk in pd.read_csv(input_file, chunksize=500000):
        chunk.iloc[:, 8] = pd.to_numeric(chunk.iloc[:, 8].replace({0: 0, 1: 1}), errors="coerce")

        for id_val in id_list:
            rows_with_id = chunk[chunk.iloc[:, 0] == id_val]

            if rows_with_id.empty:
                print(f"No lines found for drug id: {id_val}")
                continue

            for descriptor in ["IN", "PR", "RT"]:
                rows_with_tip = rows_with_id[rows_with_id.iloc[:, 2].str.strip().str.upper() == descriptor]

                if rows_with_tip.empty:
                    print(f"No lines found for drug id: {id_val}, Descriptor: {descriptor}")
                    continue

                suma = rows_with_tip.iloc[:, 8].sum()

                if id_val in sum_dict[descriptor]:
                    sum_dict[descriptor][id_val] += suma
                else:
                    sum_dict[descriptor][id_val] = suma

    id_sume_df = []
    for descriptor in sum_dict:
        for id_val, suma in sum_dict[descriptor].items():
            id_sume_df.append({"ID": id_val, "Tip ID": descriptor, "Sumă": suma})

    id_sume_df = pd.DataFrame(id_sume_df)

    suma_maxima = id_sume_df["Sumă"].max()

    jumatate_suma_maxima = suma_maxima * 0.5

    sume_top_df = id_sume_df[id_sume_df["Sumă"] >= jumatate_suma_maxima]

    with pd.ExcelWriter(output_file) as writer:
        for descriptor in ["IN", "PR", "RT"]:
            filtered_df = sume_top_df[sume_top_df["Tip ID"] == descriptor]
            filtered_df.to_excel(writer, sheet_name=descriptor, index=False)
            print(f"Tip ID: {descriptor}")
            print(filtered_df[["ID", "Sumă"]].head(3))  # Afișăm primele 3
            print("\n")

    print(f"Rezultatele pentru sumele mai mari sau egale cu 50% din suma maximă au fost salvate în {output_file}")
    return

def main():
    drugcentral_df = pd.read_excel("020 - dc drugs.xlsx")

    drug_id_list = drugcentral_df.iloc[:, 0]

    input_files = ["cnn-pred.csv", "rfc-pred.csv", "mlp-pred.csv", "svc-pred.csv"]
    output_files = ["CNN_rezultate_top_50suma.xlsx", "RFC_rezultate_top_50suma.xlsx", "MLP_rezultate_top_50suma.xlsx", "SVC_rezultate_top_50suma.xlsx"]

    for input_file, output_file in zip(input_files, output_files):
        process_file(input_file, drug_id_list, output_file)

    return

main()
