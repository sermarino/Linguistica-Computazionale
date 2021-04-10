# -*- coding: utf-8 -*-
import sys
import codecs
import nltk

def TrovaToken(frasi):
    tokenTOT = []
    for frase in frasi:
        # divido la frase in token
        tokens = nltk.word_tokenize(frase)
        for tok in tokens:
            tokenTOT.append(tok)
    return tokenTOT



def main(testo1, testo2):
    listaSostantivi     = ["NN", "NNS", "NNP", "NNPS"]
    listaVerbi          = ["VBZ", "VBD", "VB", "VBN", "VBP", "VBG"]
    listaAvverbi        = ["RB","RBR","RBS"]
    listaAggettivi      = ["JJ","JJR","JJS"]

    inputTesto1 = codecs.open(testo1, "r", "utf-8")
    inputTesto2 = codecs.open(testo2, "r", "utf-8")
    raw1 = inputTesto1.read()
    raw2 = inputTesto2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)


    #con questa  parte restituisco il numero di token e il numero di frasi per ogni testo
    lunghezzaTOT1 = 0.0
    for frase1 in frasi1:
        #divido la frase in token
        tokens = nltk.word_tokenize(frase1)
        #cacolo la lunghezza
        lunghezzaTOT1 = lunghezzaTOT1+len(tokens)
    #restituisco il risultato
    lunghezzaTOT2 = 0.0
    for frase2 in frasi2:
        # divido la frase in token
        tokens = nltk.word_tokenize(frase2)
        # cacolo la lunghezza
        lunghezzaTOT2 = lunghezzaTOT2 + len(tokens)
    # restituisco il risultato:
    #numero frasi
    print("Il file ", inputTesto1, "contiene ", len(frasi1), "frasi")
    print("Il file ", inputTesto2, "contiene ", len(frasi2), "frasi")
    if len(frasi1) > len(frasi2):
        print("Il file ", inputTesto1, " contiene più frasi del file", inputTesto2)
    elif len(frasi2) > len(frasi1):
        print("Il file ", inputTesto2, " contiene più frasi del file", inputTesto1)
    else:
        print("I due file hanno lo stesso numero di frasi")

    #TOKEN PER FRASE
    #con questa parte restituisco il numero medio di token per frase
    lunghezzaTOT1 = 0.0
    for frase1 in frasi1:
        # divido la frase in token
        tokens = nltk.word_tokenize(frase1)
        # cacolo la lunghezza
        lunghezzaTOT1 = lunghezzaTOT1 + len(tokens)
    # restituisco il risultato
    lunghezzaTOT2 = 0.0
    for frase2 in frasi2:
        # divido la frase in token
        tokens = nltk.word_tokenize(frase2)
        # cacolo la lunghezza
        lunghezzaTOT2 = lunghezzaTOT2 + len(tokens)

    numeroFrasi1 = len(frasi1)
    numeroFrasi2 = len(frasi2)
    print("La lunghezza media delle frasi in termini di toke è ",lunghezzaTOT1/numeroFrasi1," nel testo ",testo1)
    print("La lunghezza media delle frasi in termini di toke è ",lunghezzaTOT2/numeroFrasi2," nel testo ",testo2)


    #creo 2 liste che contengono i corpora, così non devo ricalcolarli ogni volta
    tokenTOT1 = []
    #scorro ogni frase del testo
    for frase1 in frasi1:
        # divido la frase in token
        tokens1 = nltk.word_tokenize(frase1)
        for tok1 in tokens1:
            #aggiungo ogni token della frase alla lista
            tokenTOT1.append(tok1)

    tokenTOT2 = []
    #scorro ogni frase del testo
    for frase2 in frasi2:
        # divido la frase in token
        tokens2 = nltk.word_tokenize(frase2)
        for tok2 in tokens2:
            #aggiungo ogni token della frase alla lista
            tokenTOT2.append(tok2)

    #NUMERO MEDIO CARATTERI
    #con questa funzione restituisco il numero medio di caratteri per parola
    numeroCaratteri1 = 0.0
    for parola1 in tokenTOT1:
        numeroCaratteri1 = numeroCaratteri1 + len(parola1)

    numeroCaratteri2 = 0.0
    for parola2 in tokenTOT2:
        numeroCaratteri2 = numeroCaratteri2 + len(parola2)
    print("La lunghezza media termini di caratteri di", testo1, " è: ", numeroCaratteri1 / len(tokenTOT1))
    print("La lunghezza media termini di caratteri di", testo2, " è: ", numeroCaratteri2 / len(tokenTOT2))


    #TTR
    #con questa funzione restituisico la TTR per ogni testo
    tokenTOT1 = TrovaToken(frasi1)
    cinquemilaToken1 = []
    numeroToken1 = 0
    while (numeroToken1 < 5001):
        cinquemilaToken1.append(tokenTOT1[numeroToken1])
        numeroToken1 += 1
    tokenTOT2 = TrovaToken(frasi2)
    cinquemilaToken2 = []
    numeroToken2 = 0
    while (numeroToken2 < 5001):
        cinquemilaToken2.append(tokenTOT2[numeroToken2])
        numeroToken2 += 1
    vocabolario1 = set(cinquemilaToken1)
    vocabolario2 = set(cinquemilaToken2)
    # TTR = C/V
    # C = lunghezza corpus; V = lunghezza vocabolario
    ttr1 = 5000 / len(vocabolario1)
    ttr2 = 5000 / len(vocabolario2)
    print("In ", testo1, " la TTR (Type Token Ratio) è uguale a ", ttr1)
    print("In ", testo2, " la TTR (Type Token Ratio) è uguale a ", ttr2)



    #PORZIONI INCREMENTALI
    #con questa parte restituisco le classi di frequenza V1,V5,V10 per porzioni incremenatali di 500 token
    print("\nNel testo",testo1," l'elenco delle classi di frequenza V1,V5,v10, per porzioni incrementali di 500 è:")
    porzioneIncrementale1       = []
    indicePorzioneIncrementale1 = 500
    while (indicePorzioneIncrementale1 < len(tokenTOT1)):
        porzioneDiVocabolario1  = []
        porzioniDiToken1        = []
        for porzione1 in range(indicePorzioneIncrementale1):
            porzioniDiToken1.append(tokenTOT1[porzione1])
        porzioneDiVocabolario1 = set(porzioniDiToken1)
        classeV1 = 0.0
        classeV5 = 0.0
        classeV10 = 0.0
        for tok in porzioneDiVocabolario1:
            if porzioniDiToken1.count(tok) == 1:
                classeV1 = classeV1 + 1
            if porzioniDiToken1.count(tok) == 5:
                classeV5 = classeV5 + 1
            if porzioniDiToken1.count(tok) == 10:
                classeV10 = classeV10 + 1
        print("In ",indicePorzioneIncrementale1,"tokens il testo ",testo1," ci sono ",classeV1,"elementi nella classe di frequenza V1")
        print("In ", indicePorzioneIncrementale1, "tokens il testo ", testo1, " ci sono ", classeV5,"elementi nella classe di frequenza V5")
        print("In ", indicePorzioneIncrementale1, "tokens il testo ", testo1, " ci sono ", classeV10,"elementi nella classe di frequenza V10")
        indicePorzioneIncrementale1 = indicePorzioneIncrementale1 + 500

    print("\nNel testo", testo2, " l'elenco delle classi di frequenza V1,V5,v10, per porzioni incrementali di 500 è:")
    porzioneIncrementale2       = []
    indicePorzioneIncrementale2 = 500
    while (indicePorzioneIncrementale2 < len(tokenTOT2)):
        porzioneDiVocabolario2  = []
        porzioniDiToken2        = []
        for porzione2 in range(indicePorzioneIncrementale2):
            porzioniDiToken2.append(tokenTOT2[porzione2])
        porzioneDiVocabolario2 = set(porzioniDiToken2)
        classeV1    = 0.0
        classeV5    = 0.0
        classeV10   = 0.0
        for tok in porzioneDiVocabolario2:
            if porzioniDiToken2.count(tok) == 1:
                classeV1 = classeV1 + 1
            if porzioniDiToken2.count(tok) == 5:
                classeV5 = classeV5 + 1
            if porzioniDiToken2.count(tok) == 10:
                classeV10 = classeV10 + 1
        print("In ", indicePorzioneIncrementale2, "tokens il testo ", testo2, " ci sono ", classeV1,
              "elementi nella classe di frequenza V1")
        print("In ", indicePorzioneIncrementale2, "tokens il testo ", testo2, " ci sono ", classeV5,
              "elementi nella classe di frequenza V5")
        print("In ", indicePorzioneIncrementale2, "tokens il testo ", testo2, " ci sono ", classeV10,
              "elementi nella classe di frequenza V10")
        indicePorzioneIncrementale2 = indicePorzioneIncrementale2 + 500


    #MEDIA SOSTANTIVI E VERBI
    #con questa parte restituiscoi il numero medio di sostantivi e di verbi per frase
    tokenPOSTesto1      = nltk.pos_tag(tokenTOT1)
    numeroSostantivi1   = 0.0
    numeroVerbi1        = 0.0
    for (token,pos) in tokenPOSTesto1:
        if pos in listaSostantivi:
            numeroSostantivi1   = numeroSostantivi1 + 1
        if pos in listaVerbi:
            numeroVerbi1        = numeroVerbi1 + 1
    tokenPOSTesto2      = nltk.pos_tag(tokenTOT2)
    numeroSostantivi2   = 0.0
    numeroVerbi2        = 0.0
    for (token, pos) in tokenPOSTesto2:
        if pos in listaSostantivi:
            numeroSostantivi2 = numeroSostantivi2 + 1
        if pos in listaVerbi:
            numeroVerbi2 = numeroVerbi2 + 1
    print("\nMedia sostantivi per frase")
    print("Nel testo",testo1,"la media dei sostantivi per frase è ",numeroSostantivi1/len(frasi1))
    print("Nel testo",testo2,"la media dei sostantivi per frase è ",numeroSostantivi2/len(frasi2))
    print("\nMedia verbi per frase")
    print("Nel testo", testo1, "la media dei verbi per frase è ", numeroVerbi1 / len(frasi1))
    print("Nel testo", testo2, "la media dei verbi per frase è ", numeroVerbi2 / len(frasi2))




    tokenPOSTesto1      = nltk.pos_tag(tokenTOT1)
    numeroSostantivi1   = 0.0
    numeroVerbi1        = 0.0
    numeroAvverbi1      = 0.0
    numeroAggettivi1    = 0.0
    numeroPuntiVirgole1 = 0.0
    for (token,pos) in tokenPOSTesto1:
        if pos in listaSostantivi:
            numeroSostantivi1   = numeroSostantivi1 + 1
        if pos in listaVerbi:
            numeroVerbi1        = numeroVerbi1 + 1
        if pos in listaAvverbi:
            numeroAvverbi1      = numeroAvverbi1 + 1
        if pos in listaAggettivi:
            numeroAggettivi1    = numeroAggettivi1 + 1
        if pos == "." or pos == ",":
            numeroPuntiVirgole1 = numeroPuntiVirgole1 +1

    tokenPOSTesto2      = nltk.pos_tag(tokenTOT2)
    numeroSostantivi2   = 0.0
    numeroVerbi2        = 0.0
    numeroAvverbi2      = 0.0
    numeroAggettivi2    = 0.0
    numeroPuntiVirgole2 = 0.0
    for (token, pos) in tokenPOSTesto2:
        if pos in listaSostantivi:
            numeroSostantivi2 = numeroSostantivi2 + 1
        if pos in listaVerbi:
            numeroVerbi2 = numeroVerbi2 + 1
        if pos in listaAvverbi:
            numeroAvverbi2 = numeroAvverbi2 + 1
        if pos in listaAggettivi:
            numeroAggettivi2 = numeroAggettivi2 + 1
        if pos == "." or pos == ",":
            numeroPuntiVirgole2 = numeroPuntiVirgole2 + 1
    print("\n")
    print("Densità lessicale del testo",testo1," ->",(numeroAggettivi1+numeroAvverbi1+numeroSostantivi1+numeroVerbi1)/(len(tokenTOT1)-numeroPuntiVirgole1))
    print("Densità lessicale del testo", testo2, " ->",(numeroAggettivi2 + numeroAvverbi2 + numeroSostantivi2 + numeroVerbi2) / (len(tokenTOT2) - numeroPuntiVirgole2))


main(sys.argv[1],sys.argv[2])