import sys
import nltk
import codecs
import math
from nltk import word_tokenize
from nltk import pos_tag
from nltk import bigrams
from nltk import FreqDist

#con questa funzione cerco, trovo e restituisco ogni bigramma per cui ogni token che lo compone è presente almeno 3 volte nel testo
def CercaTokenMaggioreDiTre(tokenTOT):
    elencoBigrammi              = bigrams(tokenTOT)
    frequenzaToken              = FreqDist(tokenTOT)
    elencoBigrammiMaggioriDiTre = []
    for (token1,token2) in elencoBigrammi:
        if frequenzaToken[token1] > 2 and frequenzaToken[token2] > 2:
            bigramma = token1,token2
            elencoBigrammiMaggioriDiTre.append(bigramma)
    return  elencoBigrammiMaggioriDiTre


#funzione principale
def main(testo1, testo2):
    # lista tag che rappresentano i sostantivi,i verbi e gli aggettivi
    listaSostantivi     = ["NN", "NNS", "NNP", "NNPS"]
    listaVerbi          = ["VBZ", "VBD", "VB", "VBN", "VBP", "VBG"]
    listaAggettivi      = ["JJ", "JJR", "JJS"]

    #leggo i file di testo
    inputTesto1 = codecs.open(testo1, "r", "utf-8")
    inputTesto2 = codecs.open(testo2, "r", "utf-8")
    raw1 = inputTesto1.read()
    raw2 = inputTesto2.read()
    #carico il tokenizzatore
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #trovo le frasi
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)

    # creo 2 liste che contengono i corpora, così non devo ricalcolarli ogni volta
    tokenTOT1 = []
    # scorro ogni frase del testo
    for frase1 in frasi1:
        # divido la frase in token
        tokens1 = word_tokenize(frase1)
        for tok1 in tokens1:
            # aggiungo ogni token della frase alla lista
            tokenTOT1.append(tok1)

    tokenTOT2 = []
    # scorro ogni frase del testo
    for frase2 in frasi2:
        # divido la frase in token
        tokens2 = word_tokenize(frase2)
        for tok2 in tokens2:
            # aggiungo ogni token della frase alla lista
            tokenTOT2.append(tok2)

    #10 POS PIU USATE
    #con questa parte restituisco le 10 pos più usatE
    elencoTokenPOSTesto1 = pos_tag(tokenTOT1)
    # salvo in un secondo elenco solo i tag di pos
    elencoPOSTesto1 = []
    for (token, pos) in elencoTokenPOSTesto1:
        elencoPOSTesto1.append(pos)
    # calcolo la distribuzione di frequenza
    frequenzaPOSTesto1 = FreqDist(elencoPOSTesto1)
    elencoDiecoPosTesto1 = frequenzaPOSTesto1.most_common(10)
    print("Per il testo", testo1, " le 10 POS (Part Of Speech) più usate sono")
    for (pos, frequenza) in elencoDiecoPosTesto1:
        print("-", pos, " che ricorre", frequenza, "volte")
    print("\n")
    elencoTokenPOSTesto2 = pos_tag(tokenTOT2)
    # salvo in un secondo elenco solo i tag di pos
    elencoPOSTesto2 = []
    for (token, pos) in elencoTokenPOSTesto2:
        elencoPOSTesto2.append(pos)
    # calcolo la distribuzione di frequenza
    frequenzaPOSTesto2 = FreqDist(elencoPOSTesto2)
    elencoDiecoPosTesto2 = frequenzaPOSTesto2.most_common(10)
    print("Per il testo", testo2, " le 10 POS (Part Of Speech) più usate sono")
    for (pos, frequenza) in elencoDiecoPosTesto2:
        print("-", pos, " che ricorre", frequenza, "volte")


    #20 SOSTANTIVI E 20 VERBI PIU USATI
    #con questa parte restituisco i 20 sostantivi e i 20 verbi più usati
    # lista tag che rappresentano i sostantivi e i verbi
    listaSostantivi = ["NN", "NNS", "NNP", "NNPS"]
    listaVerbi = ["VBZ", "VBD", "VB", "VBN", "VBP", "VBG"]
    # cero l'lenco token pos
    tokenPOSTesto1 = pos_tag(tokenTOT1)
    # elenchi vuoti
    elencoSostantiviTesto1  = []
    elencoVerbiTesto1       = []
    for (token, pos) in tokenPOSTesto1:
        if pos in listaSostantivi:
            elencoSostantiviTesto1.append(token)
        if pos in listaVerbi:
            elencoVerbiTesto1.append(token)
    # distribuzioni di frequenza e 20 più usati
    # sostantivi
    frequenzaSostantiviTesto1 = FreqDist(elencoSostantiviTesto1)
    elencoVentiSostantiviTesto1 = frequenzaSostantiviTesto1.most_common(20)
    # verbi
    frequenzaVerbiTesto1 = FreqDist(elencoVerbiTesto1)
    elencoVentiVerbiTesto1 = frequenzaVerbiTesto1.most_common(20)
    print("\n")
    print("Per il testo", testo1, " i 20 sostantivi più usati sono")
    for (sostantivo, frequenza) in elencoVentiSostantiviTesto1:
        print("-", sostantivo, "che ricorre", frequenza, "volte")
    print("------------------------------------------------")
    print("Per il testo", testo1, " i 20 verbi più usati sono")
    for (verbo, frequenza) in elencoVentiVerbiTesto1:
        print("-", verbo, "che ricorre", frequenza, "volte")

    # cero l'lenco token pos oer il testo 2
    tokenPOSTesto2 = pos_tag(tokenTOT2)
    # elenchi vuoti
    elencoSostantiviTesto2  = []
    elencoVerbiTesto2       = []
    for (token, pos) in tokenPOSTesto2:
        if pos in listaSostantivi:
            elencoSostantiviTesto2.append(token)
        if pos in listaVerbi:
            elencoVerbiTesto2.append(token)
    # distribuzioni di frequenza e 20 più usati
    # sostantivi
    frequenzaSostantiviTesto2 = FreqDist(elencoSostantiviTesto2)
    elencoVentiSostantiviTesto2 = frequenzaSostantiviTesto2.most_common(20)
    # verbi
    frequenzaVerbiTesto2 = FreqDist(elencoVerbiTesto2)
    elencoVentiVerbiTesto2 = frequenzaVerbiTesto2.most_common(20)
    print("\n")
    print("Per il testo", testo2, " i 20 sostantivi più usati sono")
    for (sostantivo, frequenza) in elencoVentiSostantiviTesto2:
        print("-", sostantivo, "che ricorre", frequenza, "volte")
    print("------------------------------------------------")
    print("Per il testo", testo2, " i 20 verbi più usati sono")
    for (verbo, frequenza) in elencoVentiVerbiTesto2:
        print("-", verbo, "che ricorre", frequenza, "volte")


    # con questa parte restituisco:
        # i 20 bigrammi formati da sostantivo e verbo
        # i 20 bigrammi formati da aggettivo e sostantivo

    tokenPOSTesto1                  = pos_tag(tokenTOT1)  # cero l'elenco token pos
    bigrammiTesto1                  = bigrams(tokenTOT1)  # elenco bigrammi
    # elenchi vuoti
    elencoSostantiviTesto1          = []
    elencoVerbiTesto1               = []
    elencoAggettiviTesto1           = []
    elencoSostantiviVerbiTesto1     = []
    elencoAggettiviSostantiviTesto1 = []
    # cerco i sostantivi, i verbi  e gli aggettvi presenti nel testo
    for (token, pos) in tokenPOSTesto1:
        if pos in listaSostantivi:
            elencoSostantiviTesto1.append(token)
        if pos in listaVerbi:
            elencoVerbiTesto1.append(token)
        if pos in listaAggettivi:
            elencoAggettiviTesto1.append(token)
    # verifico che la condizione della coppia sia corretta
    # prendo solo i bigrammi che corrispondono alle specifiche
    # in questo caso verifico che sia un sostantivo seguito da un verbo
    for (tok1, tok2) in bigrammiTesto1:
        if tok1 in elencoSostantiviTesto1 and tok2 in elencoVerbiTesto1:
            bigramma = tok1, tok2
            elencoSostantiviVerbiTesto1.append(bigramma)
        # in questo caso verifico che sia un aggettivo seguito da un sostantivo
        if tok1 in elencoAggettiviTesto1 and tok2 in elencoSostantiviTesto1:
            bigramma = tok1, tok2
            elencoAggettiviSostantiviTesto1.append(bigramma)
    # distribuzioni di frequenza e 20 più usati
    # sostantivi - verbi
    frequenzaSostantiviVerbiTesto1 = FreqDist(elencoSostantiviVerbiTesto1)
    elencoVentiSostantiviVerbiTesto1 = frequenzaSostantiviVerbiTesto1.most_common(20)
    # aggettivi - sostantivi
    frequenzaAggettiviSostantiviTesto1 = FreqDist(elencoAggettiviSostantiviTesto1)
    elencoVentiAggettiviSostantiviTesto1 = frequenzaAggettiviSostantiviTesto1.most_common(20)
    print("\n")
    print("Per il testo", testo1, "i 20 bigrammi composti da sostantivo e verbo più usati sono ")
    for ((sostantivo, verbo), frequenza) in elencoVentiSostantiviVerbiTesto1:
        print("- sostantivo:", sostantivo, "; verbo:", verbo, " Ricorrono insieme", frequenza, "volte")

    tokenPOSTesto2                  = pos_tag(tokenTOT2)  # cero l'elenco token pos
    bigrammiTesto2                  = bigrams(tokenTOT2)  # elenco bigrammi
    # elenchi vuoti
    elencoSostantiviTesto2          = []
    elencoVerbiTesto2               = []
    elencoAggettiviTesto2           = []
    elencoSostantiviVerbiTesto2     = []
    elencoAggettiviSostantiviTesto2 = []
    # cerco i sostantivi, i verbi e gli aggettivi apresenti nel testo
    for (token, pos) in tokenPOSTesto2:
        if pos in listaSostantivi:
            elencoSostantiviTesto2.append(token)
        if pos in listaVerbi:
            elencoVerbiTesto2.append(token)
        if pos in listaAggettivi:
            elencoAggettiviTesto2.append(token)
    # verifico che la condizione della coppia sia corretta
    # prendo solo i bigrammi che corrispondono alle specifiche
    # in questo caso verifico che sia un sostantivo seguito da un verbo
    for (tok1, tok2) in bigrammiTesto2:
        if tok1 in elencoSostantiviTesto2 and tok2 in elencoVerbiTesto2:
            bigramma = tok1, tok2
            elencoSostantiviVerbiTesto2.append(bigramma)
        # in questo caso verifico che sia un aggettivo seguito da un sostantivo
        if tok1 in elencoAggettiviTesto2 and tok2 in elencoSostantiviTesto2:
            bigramma = tok1, tok2
            elencoAggettiviSostantiviTesto2.append(bigramma)
    # distribuzioni di frequenza e 20 più usati
    # sostantivi - verbi
    frequenzaSostantiviVerbiTesto2          = FreqDist(elencoSostantiviVerbiTesto2)
    elencoVentiSostantiviVerbiTesto2        = frequenzaSostantiviVerbiTesto2.most_common(20)
    # aggettivi - sostantivi
    frequenzaAggettiviSostantiviTesto2      = FreqDist(elencoAggettiviSostantiviTesto2)
    elencoVentiAggettiviSostantiviTesto2    = frequenzaAggettiviSostantiviTesto2.most_common(20)
    print("\n")
    print("Per il testo", testo2, "i 20 bigrammi composti da sostantivo e verbo più usati sono ")
    for ((sostantivo, verbo), frequenza) in elencoVentiSostantiviVerbiTesto2:
        print("- sostantivo:", sostantivo, "; verbo:", verbo, " Ricorrono insieme", frequenza, "volte")

    # adesso stampo i bigrammi composti da aggettivi e sostantivi
    print("\n")
    print("Per il testo", testo1, "i 20 bigrammi composti da aggettivo e  sostantivo più usati sono ")
    for ((aggettivo, sostantivo), frequenza) in elencoVentiAggettiviSostantiviTesto1:
        print("- aggettivo:", aggettivo, "; sostantivo:", sostantivo, " Ricorrono insieme", frequenza, "volte")

    print("\n")
    print("Per il testo", testo2, "i 20 bigrammi composti da aggettivo e  sostantivo più usati sono ")
    for ((aggettivo, sostantivo), frequenza) in elencoVentiAggettiviSostantiviTesto2:
        print("- aggettivo:", aggettivo, "; sostantivo:", sostantivo, " Ricorrono insieme", frequenza, "volte")

    #PROBABILITA CONGIUNTA E PROBABILITA CONDIZIONATA
    #con questa parte restituisco la probabilità congiunta e quella condizionata
    # per la probabilità condizionat e congiunta a ho bisogno delle frequenze dei tokens e dei bigrammi che compongono il testo
    lunghezzaTesto1                             = len(tokenTOT1)
    elencoBigrammiTesto1                        = CercaTokenMaggioreDiTre(tokenTOT1)
    elencoBigrammiDiversiTesto1                 = set(elencoBigrammiTesto1)
    frequenzaTokenTesto1                        = FreqDist(tokenTOT1) #questa distribuzione viene richiamata più volte durante il programma
    frequenzaBigrammiTesto1                     = FreqDist(elencoBigrammiTesto1) #questa distribuzione viene richiamata più volte durante il programma
    elencoBigrammiFrequenzeCondizionateTesto1   = []
    elencoBigrammiFrequenzeCongiunteTesto1      = []

    for (token1, token2) in elencoBigrammiDiversiTesto1:
        # trovo tutte le frequenze che mi servono e poi le aggiungo alla lista
        bigramma                = token1, token2
        frequenzaBigramma       = frequenzaBigrammiTesto1[bigramma]
        frequenzaToken1         = frequenzaTokenTesto1[token1]
        probabilitaCondizionata = (frequenzaBigramma * 1.0) / (frequenzaToken1 * 1.0)
        # per la probabilità congiunta ho bisongo di sapere la probabilità condizionata
        # quindi posso crere questo elenco anche ora
        probabilitaCongiunta = (frequenzaToken1 / lunghezzaTesto1) * probabilitaCondizionata
        elencoBigrammiFrequenzeCondizionateTesto1.append((bigramma, probabilitaCondizionata))
        elencoBigrammiFrequenzeCongiunteTesto1.append((bigramma, probabilitaCongiunta))
    # ora devo trovare i primi 20
    # uso sorted per riodinare gli elenchi
    elencoDiscendenteBigrammiFrequenzeCondizionateTesto1 = sorted(elencoBigrammiFrequenzeCondizionateTesto1,
                                                                  key=lambda a: -a[1], reverse=False)
    elencoDiscendenteBigrammiFrequenzeCongiunteTesto1 = sorted(elencoBigrammiFrequenzeCongiunteTesto1,
                                                               key=lambda a: -a[1], reverse=False)
    # stampo i risultati
    print("\n")
    print("Per il testo", testo1, "i 20 bigrammi con probabilità Condizionata massima sono ")
    for ((token1, token2), probabilitaCondizionata) in elencoDiscendenteBigrammiFrequenzeCondizionateTesto1[
                                                       :20]:  # con [:20] prendo soltanto i primi 20 elementi
        print("- token1:", token1, "; token2:", token2, "hanno una probablità condizionata massima di ",
              probabilitaCondizionata)
    print("\n")
    print("Per il testo", testo1, "i 20 bigrammi con probabilità congiunta massima sono ")
    for ((token1, token2), probabilitaCongiunta) in elencoDiscendenteBigrammiFrequenzeCongiunteTesto1[
                                                    :20]:  # con [:20] prendo soltanto i primi 20 elementi
        print("- token1:", token1, "; token2:", token2, "hanno una probablità condizionata massima di ",
              probabilitaCongiunta)

    # testo2
    # per la probabilità condizionat e congiunta a ho bisogno delle frequenze dei tokens e dei bigrammi che compongono il testo
    lunghezzaTesto2                             = len(tokenTOT2)
    elencoBigrammiTesto2                        = CercaTokenMaggioreDiTre(tokenTOT2)
    elencoBigrammiDiversiTesto2                 = set(elencoBigrammiTesto2)
    frequenzaTokenTesto2                        = FreqDist(tokenTOT2) #questa distribuzione viene richiamata più volte durante il programma
    frequenzaBigrammiTesto2                     = FreqDist(elencoBigrammiTesto2) #questa distribuzione viene richiamata più volte durante il programma
    elencoBigrammiFrequenzeCondizionateTesto2   = []
    elencoBigrammiFrequenzeCongiunteTesto2      = []

    for (token1, token2) in elencoBigrammiDiversiTesto2:
        # trovo tutte le frequenze che mi servono e poi le aggiungo alla lista
        bigramma = token1, token2
        frequenzaBigramma = frequenzaBigrammiTesto2[bigramma]
        frequenzaToken1 = frequenzaTokenTesto2[token1]
        probabilitaCondizionata = (frequenzaBigramma * 1.0) / (frequenzaToken1 * 1.0)
        # per la probabilità congiunta ho bisongo di sapere la probabilità condizionata
        # quindi posso crere questo elenco anche ora
        probabilitaCongiunta = (frequenzaToken1 / lunghezzaTesto2) * probabilitaCondizionata
        elencoBigrammiFrequenzeCondizionateTesto2.append((bigramma, probabilitaCondizionata))
        elencoBigrammiFrequenzeCongiunteTesto2.append((bigramma, probabilitaCongiunta))
    # ora devo trovare i primi 20
    # uso sorted per riodinare gli elenchi
    elencoDiscendenteBigrammiFrequenzeCondizionateTesto2 = sorted(elencoBigrammiFrequenzeCondizionateTesto2,
                                                                  key=lambda a: -a[1], reverse=False)
    elencoDiscendenteBigrammiFrequenzeCongiunteTesto2 = sorted(elencoBigrammiFrequenzeCongiunteTesto2,
                                                               key=lambda a: -a[1], reverse=False)
    # stampo i risultati
    print("\n")
    print("Per il testo", testo2, "i 20 bigrammi con probabilità Condizionata massima sono ")
    for ((token1, token2), probabilitaCondizionata) in elencoDiscendenteBigrammiFrequenzeCondizionateTesto2[
                                                       :20]:  # con [:20] prendo soltanto i primi 20 elementi
        print("- token1:", token1, "; token2:", token2, "; Hanno una probablità condizionata massima di ",
              probabilitaCondizionata)
    print("\n")
    print("Per il testo", testo2, "i 20 bigrammi con probabilità congiunta massima sono ")
    for ((token1, token2), probabilitaCongiunta) in elencoDiscendenteBigrammiFrequenzeCongiunteTesto2[
                                                    :20]:  # con [:20] prendo soltanto i primi 20 elementi
        print("- token1:", token1, "; token2:", token2, "; Hanno una probablità condizionata massima di ",
              probabilitaCongiunta)



    #LOCAL MUTUAL INFORMATION
    #con questa parte restituisco la Lcoal Mutual information di 20 bigrammi
    # qui ci sono tutte le info che mi serviranno dopo per il calcolo
    lunghezzaTesto1             = len(tokenTOT1)
    elencoBigrammiTesto1        = CercaTokenMaggioreDiTre(tokenTOT1)
    elencoBigrammiDiversiTesto1 = set(elencoBigrammiTesto1)
    elencoLMItesto1             = []
    # scorro i bigrammi del testo
    # trovo la frequenza di token1, token2 e del bigramma
    for (token1, token2) in elencoBigrammiDiversiTesto1:
        probabilitaToken1 = frequenzaTokenTesto1[token1] * 1.0 / lunghezzaTesto1 * 1.0
        probabilitaToken2 = frequenzaTokenTesto1[token2] * 1.0 / lunghezzaTesto1 * 1.0
        probabilitaBigramma = frequenzaBigrammiTesto1[(token1, token2)] * 1.0 / lunghezzaTesto1 * 1.0
        # calcolo local mutual information
        localMutualInformation = (probabilitaBigramma) * (
            math.log(probabilitaBigramma / (probabilitaToken1 * probabilitaToken2)))
        elencoLMItesto1.append(((token1, token2), localMutualInformation))
    # riodino in ordine decrescente
    elencoDiscendenteLMItesto1 = sorted(elencoLMItesto1, key=lambda a: -a[1], reverse=False)
    print("\n")
    print("Per il testo", testo1, "i 20 bigrammi con probabilità Local Mutual Information massima sono ")
    for ((token1, token2), lmi) in elencoDiscendenteLMItesto1[:20]:  # con [:20] prendo soltanto i primi 20 elementi
        print("- token1:", token1, "; token2:", token2, "; Hanno una Local Mutual Information di ", lmi)
    # secondo testo
    # qui ci sono tutte le info che mi serviranno dopo per il calcolo
    lunghezzaTesto2             = len(tokenTOT2)
    elencoBigrammiTesto2        = CercaTokenMaggioreDiTre(tokenTOT2)
    elencoBigrammiDiversiTesto2 = set(elencoBigrammiTesto2)
    elencoLMItesto2             = []
    # scorro i bigrammi del testo
    # trovo la frequenza di token1, token2 e del bigramma
    for (token1, token2) in elencoBigrammiDiversiTesto2:
        probabilitaToken1 = frequenzaTokenTesto2[token1] * 1.0 / lunghezzaTesto2 * 1.0
        probabilitaToken2 = frequenzaTokenTesto2[token2] * 1.0 / lunghezzaTesto2 * 1.0
        probabilitaBigramma = frequenzaBigrammiTesto2[(token1, token2)] * 1.0 / lunghezzaTesto2 * 1.0
        localMutualInformation = (probabilitaBigramma) * (
            math.log(probabilitaBigramma / (probabilitaToken1 * probabilitaToken2)))
        # calcolo local mutual information
        elencoLMItesto2.append(((token1, token2), localMutualInformation))
        # riodino in ordine decrescente
    elencoDiscendenteLMItesto2 = sorted(elencoLMItesto2, key=lambda a: -a[1], reverse=False)
    print("\n")
    print("Per il testo", testo2, "i 20 bigrammi con probabilità Lcoal Mutal Information massima sono ")
    for ((token1, token2), lmi) in elencoDiscendenteLMItesto2[:20]:  # con [:20] prendo soltanto i primi 20 elementi
        print("- token1:", token1, "; token2:", token2, "; Hanno una Local Mutual Information di ", lmi)

    #MARKOV ORDINE 1
    #con  questa parte restituisco la probibilità per frasi con almeno 8 token e al massimo 15, attraverso una catena Markovian di ordine 1
    # con l'aiuto del metodo "Add-one Smoothing"
    # dizionario = {
    # 'n': [(frasi con lunghezza n,markov1)]
    # }
    dizionarioFrasiTesto1   = {}
    elencoBigrammiTesto1    = CercaTokenMaggioreDiTre(tokenTOT1)
    lunghezzaTesto1         = len(tokenTOT1)
    voabolarioTesto1        = len(set(tokenTOT1))
    for indice in range(8, 16):
        dizionarioFrasiTesto1[indice] = []
    for frase in frasi1:
        tokens = word_tokenize(frase)
        elencoBigrammiFraseTesto1 = list(bigrams(tokens))
        for tok in tokens:
            probabilita = (frequenzaTokenTesto1[tokens[0]] * 1.0) / lunghezzaTesto1
            for (token1, token2) in elencoBigrammiFraseTesto1:
                bigramma = token1, token2
                dividendo = frequenzaBigrammiTesto1[bigramma] + 1
                divisore = frequenzaTokenTesto1[token1] + voabolarioTesto1
                probabilitaBigramma = dividendo / divisore
                probabilita = probabilita * probabilitaBigramma
            # per ogni lunghezza da 8 a 15 c'è una "key" del dizionario
            # il bigramma frase-valore viene aggiunto come valore alla key corrispontente
            # il valore è una lista composta da bigrammi
            if len(tokens) == 8:
                dizionarioFrasiTesto1[8].append((frase, probabilita))
            if len(tokens) == 9:
                dizionarioFrasiTesto1[9].append((frase, probabilita))
            if len(tokens) == 10:
                dizionarioFrasiTesto1[10].append((frase, probabilita))
            if len(tokens) == 11:
                dizionarioFrasiTesto1[11].append((frase, probabilita))
            if len(tokens) == 12:
                dizionarioFrasiTesto1[12].append((frase, probabilita))
            if len(tokens) == 13:
                dizionarioFrasiTesto1[13].append((frase, probabilita))
            if len(tokens) == 14:
                dizionarioFrasiTesto1[14].append((frase, probabilita))
            if len(tokens) == 15:
                dizionarioFrasiTesto1[15].append((frase, probabilita))
        # riordino in modo tale che il valore più alto sia il primo della lista composta da bigrammi
        for indice in range(8, 16):
            sorted(dizionarioFrasiTesto1[indice], key=lambda a: -a[1], reverse=False)
    for indice in range(8, 16):
        print("\n")
        print("Per il testo", testo1, "la frase che contiene ", indice,
              "tokens con  una probabilità più alta calcolata attravero un modello di Markov di ordine 1 usando lo Add-one Smoothing è ")
        print("FRASE: ")
        print(dizionarioFrasiTesto1[indice][0][0])
        print("FINE FRASE")
        print("La frase ha probabilità: ", dizionarioFrasiTesto1[indice][0][1])

    # testo2
    # dizionario = {
    # 'n': [(frasi con lunghezza n, markov1)],
    # }
    dizionarioFrasiTesto2   = {}
    elencoBigrammiTesto2    = CercaTokenMaggioreDiTre(tokenTOT2)
    lunghezzaTesto2         = len(tokenTOT2)
    voabolarioTesto2        = len(set(tokenTOT2))
    for indice in range(8, 16):
        dizionarioFrasiTesto2[indice] = []
    for frase in frasi2:
        tokens = word_tokenize(frase)
        elencoBigrammiFraseTesto2 = list(bigrams(tokens))
        for tok in tokens:
            probabilita = (frequenzaTokenTesto2[tokens[0]] * 1.0) / lunghezzaTesto2
            for (token1, token2) in elencoBigrammiFraseTesto2:
                bigramma = token1, token2
                dividendo = frequenzaBigrammiTesto2[bigramma] + 1
                divisore = frequenzaTokenTesto2[token1] + voabolarioTesto2
                probabilitaBigramma = dividendo / divisore
                probabilita = probabilita * probabilitaBigramma
            # per ogni lunghezza da 8 a 15 c'è una "key" del dizionario
            # il bigramma frase-valore viene aggiunto come valore alla key corrispontente
            # il valore è una lista composta da bigrammi
            if len(tokens) == 8:
                dizionarioFrasiTesto2[8].append((frase, probabilita))
            if len(tokens) == 9:
                dizionarioFrasiTesto2[9].append((frase, probabilita))
            if len(tokens) == 10:
                dizionarioFrasiTesto2[10].append((frase, probabilita))
            if len(tokens) == 11:
                dizionarioFrasiTesto2[11].append((frase, probabilita))
            if len(tokens) == 12:
                dizionarioFrasiTesto2[12].append((frase, probabilita))
            if len(tokens) == 13:
                dizionarioFrasiTesto2[13].append((frase, probabilita))
            if len(tokens) == 14:
                dizionarioFrasiTesto2[14].append((frase, probabilita))
            if len(tokens) == 15:
                dizionarioFrasiTesto2[15].append((frase, probabilita))
        # riordino in modo tale che il valore più alto sia il primo della lista composta da bigrammi
        for indice in range(8, 16):
            sorted(dizionarioFrasiTesto2[indice], key=lambda a: -a[1], reverse=False)
    for indice in range(8, 16):
        print("\n")
        print("Per il testo", testo2, "la frase che contiene ", indice,
              "tokens con  una probabilità più alta calcolata attravero un modello di Markov di ordine 1 usando lo Add-one Smoothing è ")
        print("FRASE: ")
        print(dizionarioFrasiTesto2[indice][0][0])
        print("FINE FRASE")
        print("La frase ha probabilità: ", dizionarioFrasiTesto2[indice][0][1])

    #10 NOMI DI PERSONA PIU USATI E 10 NOMI DI LUOGO PIU USATI
    #con questa parte restiuisco i 10 nomi di persona e i 10 luoghi più frequenti
    NE1                     = []
    elencoPersoneTesto1     = []
    elencoGPETesto1         = []
    tokenPOSTesto1          = pos_tag(tokenTOT1)
    analisiTesto1           = nltk.ne_chunk(tokenPOSTesto1)  # rappresentazione ad albero
    IOBformat               = nltk.chunk.tree2conllstr(analisiTesto1)  # trasformo in formato IOB
    for nodo in analisiTesto1:  # ciclo l'albero scorrendo i nodi
        if hasattr(nodo, 'label'):  # controlla se e' un nodo intermedio
            if nodo.label() == "PERSON":
                elemento = nodo.leaves()
                # converte l'elemento in una tupla (utile per elementi composti da piu token che altrimenti verrebbero restituiti in sottoliste)
                elencoPersoneTesto1.append(tuple(elemento))
            if nodo.label() == "GPE":
                elementGPE = nodo.leaves()
                # converte l'elemento in una tupla (utile per elementi composti da piu token che altrimenti verrebbero restituiti in sottoliste)
                elencoGPETesto1.append(tuple(elementGPE))
    # calcolo le frequenze
    frequenzaPersoneTesto1 = FreqDist(elencoPersoneTesto1).items()
    elenccoDiscendentePersoneTesto1 = sorted(frequenzaPersoneTesto1, key=lambda a: -a[1], reverse=False)
    print("\n")
    print("Per il testo", testo1, "i 15 nomi di persona più usati sono ")
    for (persona, frequenza) in elenccoDiscendentePersoneTesto1[:15]:
        print("- Nome: ", persona, "; Ricorre ", frequenza, "volte")

    frequenzaLuoghiTesto1 = FreqDist(elencoGPETesto1).items()
    elenccoDiscendenteluoghiTesto1 = sorted(frequenzaLuoghiTesto1, key=lambda a: -a[1], reverse=False)
    print("\n")
    print("Per il testo", testo1, "i 15 nomi di luogo più usati sono ")
    for (luogo, frequenza) in elenccoDiscendenteluoghiTesto1[:15]:
        print("- Nome: ", luogo, "; Ricorre ", frequenza, "volte")

    # testo2
    NE1                     = []
    elencoPersoneTesto2     = []
    elencoGPETesto2         = []
    tokenPOSTesto2          = pos_tag(tokenTOT2)
    analisiTesto2           = nltk.ne_chunk(tokenPOSTesto2)  # rappresentazione ad albero
    IOBformat               = nltk.chunk.tree2conllstr(analisiTesto2)  # formato IOB
    for nodo in analisiTesto2:  # ciclo l'albero scorrendo i nodi
        if hasattr(nodo, 'label'):  # controlla se è un nodo intermedio
            if nodo.label() == "PERSON":
                elemento = nodo.leaves()
                # converte l'elemento in una tupla
                elencoPersoneTesto2.append(tuple(elemento))
            if nodo.label() == "GPE":
                elementGPE = nodo.leaves()
                # converte l'elemento in una tupla
                elencoGPETesto2.append(tuple(elementGPE))
    # calcolo le frequenze
    frequenzaPersoneTesto2 = FreqDist(elencoPersoneTesto2).items()
    elenccoDiscendentePersoneTesto2 = sorted(frequenzaPersoneTesto2, key=lambda a: -a[1], reverse=False)
    print("\n")
    print("Per il testo", testo2, "i 15 nomi di persona più usati sono ")
    for (persona, frequenza) in elenccoDiscendentePersoneTesto2[:15]:
        print("- Nome: ", persona, "; Ricorre ", frequenza, "volte")


    frequenzaLuoghiTesto2 = FreqDist(elencoGPETesto2).items()
    elenccoDiscendenteluoghiTesto2 = sorted(frequenzaLuoghiTesto2, key=lambda a: -a[1], reverse=False)
    print("\n")
    print("Per il testo", testo2, "i 15 nomi di luogo più usati sono ")
    for (luogo, frequenza) in elenccoDiscendenteluoghiTesto2[:15]:
        print("- Nome: ", luogo, "; Ricorre ", frequenza, "volte")

main(sys.argv[1],sys.argv[2])