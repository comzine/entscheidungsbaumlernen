import pandas as pd
import random


def generiere_kategorische_daten(anzahl_datensaetze, feature_liste):
    """
    Generiert einen DataFrame mit kategorischen Daten und explizit benannten Kategoriewerten.

    Parameter:
    - anzahl_datensaetze (int): Anzahl der Datensätze (Zeilen) im DataFrame.
    - feature_liste (list): Liste von Tuples mit Feature-Namen als erstem Element und einer Liste von Kategorienamen als zweitem Element.

    Rückgabe:
    - DataFrame: Ein pandas DataFrame mit kategorischen Daten.
    """
    # Initialisiere den DataFrame
    daten = pd.DataFrame()

    # Für jedes Feature generiere zufällige kategorische Daten
    for feature, werte in feature_liste:
        # Erzeuge zufällige Daten für jedes Feature
        daten[feature] = [random.choice(werte) for _ in range(anzahl_datensaetze)]

    return daten


def generiere_eindeutige_daten(anzahl_datensaetze, feature_liste):
    """
    Generiert einen DataFrame mit eindeutigen kategorischen Daten für Features und fügt dann das Zielmerkmal hinzu.

    Parameter:
    - anzahl_datensaetze (int): Anzahl der Datensätze (Zeilen) im DataFrame.
    - feature_liste (list): Liste von Tuples mit Feature-Namen als erstem Element und einer Liste von Kategorienamen als zweitem Element.

    Rückgabe:
    - DataFrame: Ein pandas DataFrame mit eindeutigen kategorischen Daten und Zielmerkmal.
    """
    # Das letzte Feature in der Liste wird als Zielmerkmal behandelt
    zielmerkmal_name, zielmerkmal_werte = feature_liste[-1]
    feature_liste_ohne_ziel = feature_liste[:-1]

    # Generiere zunächst Daten ohne das Zielmerkmal
    daten = generiere_kategorische_daten(anzahl_datensaetze, feature_liste_ohne_ziel)

    # Entferne Duplikate
    daten = daten.drop_duplicates().reset_index(drop=True)

    # Füge das Zielmerkmal hinzu
    daten[zielmerkmal_name] = [
        random.choice(zielmerkmal_werte) for _ in range(len(daten))
    ]

    return daten
