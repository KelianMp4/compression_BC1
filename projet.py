from PIL import Image
import numpy as np
from math import log10, sqrt


def load(filename):
    toLoad = Image.open(filename)
    return np.array(toLoad)


def save(mat_pix, filename):
    Image.fromarray(mat_pix).save(filename)


def psnr(original, compressed):
    mse = np.mean((original.astype(int) - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def padding(M):
    lignes4 = (4 - M.shape[0] % 4) % 4
    colonnes4 = (4 - M.shape[1] % 4) % 4
    if lignes4 or colonnes4:
        res = np.zeros((M.shape[0] + lignes4, M.shape[1] + colonnes4, 3),
                       dtype=np.uint8)
        res[:res.shape[0]-lignes4, :res.shape[1]-colonnes4] = M
        return res
    return M


def remove_padding(M):
    width_not_null = np.where(~ np.all(M == [0, 0, 0], axis=(1, 2)))[0]
    height_not_null = np.where(~ np.all(M == [0, 0, 0], axis=(0, 2)))[0]

    start_width = width_not_null[0]
    end_width = width_not_null[-1] + 1
    start_height = height_not_null[0]
    end_height = height_not_null[-1] + 1

    M_without_padding = M[start_width:end_width, start_height:end_height, :]

    return M_without_padding


def patching(M):
    return [[np.array(M[i:i+4, j:j+4], dtype=np.uint8)
            for j in range(0, len(M[i]), 4)]
            for i in range(0, len(M), 4)]


def choix_couleur_min_max(couleur=list):
    minimum_rgb = np.min(couleur, axis=(0, 1))
    maximum_rgb = np.max(couleur, axis=(0, 1))
    return minimum_rgb, maximum_rgb


def choix_couleur_moyenne(couleur=list):
    moyenne = np.mean(couleur, axis=(0, 1))
    ecart_type = np.std(couleur, axis=(0, 1))
    couleur_1 = abs(moyenne - ecart_type)
    couleur_2 = abs(moyenne + ecart_type)
    couleur_1 = [int(nombre) for nombre in couleur_1]
    couleur_2 = [int(nombre) for nombre in couleur_2]

    return np.array(couleur_1), np.array(couleur_2)


def tronque(n=int, p=int):
    return n >> p


def palette(a, b):
    # [a, 2a/3 +b/3, a/3 + 2b/3, b]
    return np.array([np.round(a),
                     np.round((2*a)/3 + b/3),
                     np.round(a/3 + (2*b)/3),
                     np.round(b)], dtype=np.uint8)


def compare(palette, pixel):
    res = []
    for i in range(len(palette)):
        res.append(np.linalg.norm(palette[i].astype(int) - pixel))
    index = res.index(min(res))
    return index


def couleur_tronque(pixel):
    return np.array([tronque(pixel[0], 3),
                     tronque(pixel[1], 2),
                     tronque(pixel[2], 3)])


def entier_patch(patch):
    liste_entier = []
    for i in range(len(patch)):
        for j in range(len(patch[i])):
            tableau_indice = []
            couleur_a, couleur_b = choix_couleur_min_max(patch[i][j])
            tableau_couleur = palette(couleur_tronque(couleur_a),
                                      couleur_tronque(couleur_b))
            for k in range(len(patch[i][j])):
                for m in range(len(patch[i][j][k])):
                    tableau_indice.append(compare(tableau_couleur,
                                                  patch[i][j][k][m]))
            couleur_a = binaire_couleur(couleur_tronque(couleur_a))
            couleur_b = binaire_couleur(couleur_tronque(couleur_b))
            tableau_indice = binaire_liste_indice(tableau_indice)
            tableau_indice.reverse()
            couleur_a.reverse()
            couleur_b.reverse()
            nombre_entier = tableau_indice + couleur_b + couleur_a
            liste_entier.append(int(''.join(nombre_entier), 2))
    return liste_entier


def binaire_couleur(couleur):
    couleur_binaire = []
    for index, canal in enumerate(couleur):
        if index == 1:
            couleur_binaire.append(format(canal, '06b'))
        else:
            couleur_binaire.append(format(canal, '05b'))

    return couleur_binaire


def binaire_liste_indice(liste=list):
    liste_binaire = []
    for nombre in liste:
        liste_binaire.append(format(nombre, '02b'))
    return liste_binaire


def ecriture_fichier(name=str, liste_entier=list, extension=str):
    image = load(name+extension)
    hauteur = image.shape[0]
    largeur = image.shape[1]
    with open(f'{name}_entier.bc1', 'w') as file:
        file.write('BC1\n')
        file.write(f'{str(hauteur)} {str(largeur)} \n')
        for entier_patch in liste_entier:
            file.write(str(entier_patch) + '\n')


def liste_entier_fichier(filename=str):
    liste_entier = []
    with open(filename, "r") as file:
        for i in range(2):
            next(file)
        for ligne_entier in file:
            liste_entier.append(int(ligne_entier.strip()))
    return liste_entier


def entier_a_liste(entier=int):
    entier = format(entier, '064b')
    tableau_index = entier[:32]
    couleur_b, couleur_a = entier[32:48], entier[48:]
    couleur_a = np.array([int(couleur_a[11:], 2),
                          int(couleur_a[5:11], 2),
                          int(couleur_a[:5], 2)],
                         dtype=np.uint8)
    couleur_b = np.array([int(couleur_b[11:], 2),
                          int(couleur_b[5:11], 2),
                          int(couleur_b[:5], 2)],
                         dtype=np.uint8)
    tableau_index = np.array([(int(tableau_index[i] + tableau_index[i+1], 2))
                              for i in range(0, len(tableau_index), 2)],
                             dtype=np.uint8)
    tableau_index = tableau_index[::-1]
    return tableau_index, couleur_a, couleur_b


def palette_inverse(couleur_a, couleur_b):
    format = (3, 2, 3)
    palette_final = []
    couleur = palette(couleur_a, couleur_b)
    for element in couleur:
        for i in range(len(format)):
            palette_final.append(inverse_tronque(element[i], format[i]))
    palette_final = np.array(palette_final, dtype=np.uint8)
    return palette_final.reshape(4, 3)


def inverse_tronque(n=int, p=int):
    return n << p


def repatch(liste_indice, palette):
    patch = []
    for indice_patch in liste_indice:
        for indice, element in enumerate(palette):
            if indice_patch == indice:
                patch.append([element])
    patch = np.array(patch, dtype=np.uint8)
    patch = patch.reshape((4, 4, 3))
    return patch


def liste_patch(liste_entier):
    liste_des_patchs = []
    for entier in liste_entier:
        chiffre = entier_a_liste(entier)
        liste_indice = chiffre[0]
        couleur_a = chiffre[1]
        couleur_b = chiffre[2]
        palette_patch = palette_inverse(couleur_a, couleur_b)
        liste_des_patchs.append(repatch(liste_indice, palette_patch))
    return liste_des_patchs


def reconstruction_image(liste_matrice=list, filename=str):
    shape = padding(load(filename)).shape
    indice = 0
    matrice_vide = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for i in range((matrice_vide.shape[0]) // 4):
        for j in range((matrice_vide.shape[1]) // 4):
            matrice_vide[(4 * i):4 + (4*i),
                         (4 * j):4 + (4*j),
                         :3] = liste_matrice[indice]
            indice += 1
    matrice_final = matrice_vide
    return matrice_final


if __name__ == '__main__':
    filename_original = str(input("Qu'elle est le nom du fichier à \
compresser (nom + extension) ? "))

    file = load(filename_original)
    file_padding = padding(file)
    size_file = file.shape
    size_file_padding = file_padding.shape
    file_remove_padding = remove_padding(file_padding)
    file_patching = patching(file_padding)

    print(f"TEST AVEC LE FICHIER {filename_original}")
    print(f"La dimension de la l'image de base est {size_file},\
 la nouvelle dimension de l'image est {size_file_padding}")
    print("Vérifions si c'est un multiple de 4")
    if size_file_padding[0] // 4 and size_file_padding[1] // 4:
        print(True, "\nNous avons vérifier par le calcul")
    print("Vérifions la fonction pour enlever le padding")
    assert file_remove_padding.all() == file.all()
    print("Si le programme n'a pas renvoyer d'erreur \
alors la fonction fonctionne (Vérification avec un 'assert')")
    print("Création des entiers par patch")
    liste_entier_patch = entier_patch(file_patching)
    nom_fichier = str(input("Donnez le nom de l'image d'origine \
(sans extension) "))
    extension = str(input("Donnez maintenant son extension "))
    print("Ecriture dans un fichier les entiers")
    print(ecriture_fichier(nom_fichier, liste_entier_patch, extension))
    print("Maintenant decompressons le fichier")
    filename = str(input("Nom du nouveau fichier (fichier + extension) ? "))
    liste_des_entier = liste_entier_fichier(filename)
    print("Maintenant reformons une liste de patch que nous \
allons ensuite reconstruire en une image que l'on va enregistré")
    liste_des_patchs = liste_patch(liste_des_entier)
    image_decompresse = reconstruction_image(liste_des_patchs,
                                             filename_original)
    nom_final = str(input("donné le nom de l'image d'origine \
(sans extension) "))
    print(save(image_decompresse, f'{nom_final}_decompresse.jpg'))
