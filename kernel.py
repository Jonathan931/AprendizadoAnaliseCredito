# -*- coding: utf-8 -*-
import pandas as pd;

def carregarDados():
    dados = pd.read_csv('./dataset/BNG(credit-g).csv');
    x = dados.drop('Class', axis=1);

    #checking status lógico 
    x['cs'] = 0;
    #credit history lógico
    x['ch'] = 0;
    #purpose lógico
    x['pp'] = 0;
    #credit_amount lógico
    x['ca'] = 0;			
    #savings status lógico
    x['ss'] = 0;
    #employment lógico
    x['ep'] = 0;
    #personal status lógico
    x['ps'] = 0;
    #residence since lógico
    x['rs'] = 0;
    #property magnitude lógico
    x['pm'] = 0;
    #housing lógico
    x['hs'] = 0;
    #existing credits lógico
    x['ec'] = 0;
    #job lógico
    x['jo'] = 0;
    #num_dependents lógico
    x['nd'] = 0;
    #own_telephone lógico
    x['ot'] = 0;
    #foreign_worker lógico
    x['fw'] = 0;

    for i, value in enumerate(x):

        if ( x['checking_status'][i] <= 2):
            x['cs'][i] = 1;

        if ( x['credit_history'][i] <= 2 ):
            x['ch'][i] = 1;

        if ( x['purpose'][i] <= 3 ):
            x['pp'][i] = 1;

        if ( x['credit_amount'][i] >= 5000):
	       x['ca'][i] = 1;
 	
        if ( x['savings_status'][i] <= 1):
            x['ss'][i]= 1;

        if ( x['employment'][i] >= 2):
            x['ep'][i]= 1;	

        if ( x['personal_status'][i] >= 1):
            x['ps'][i]= 1; 

        if ( x['residence_since'][i] >= 2):
            x['rs'][i]= 1; 

        if ( x['property_magnitude'][i] <= 1):
            x['pm'][i]= 1;

        if ( x['housing'][i] <= 1):
            x['hs'][i]= 1;

        if ( x['existing_credits'][i] <= 1):
            x['ec'][i]= 1; 

        if ( x['job'][i] >= 2):
            x['jo'][i]= 1;

        if ( x['num_dependents'][i] <= 1):
            x['nd'][i]= 1; 

        if ( x['own_telephone'][i] >= 2):
            x['ot'][i]= 1;

        if ( x['foreign_worker'][i] == 1):
            x['fw'][i]= 1; 


    y = dados.Class;

    print(x);
    
   
    return x, y;


def main():
    dados, resultados = carregarDados();

    print( "Dados carregados...." )

    from sklearn.model_selection import train_test_split
    treino_dados, teste_dados, treino_resultados, teste_resultados = train_test_split(dados, resultados, test_size=0.1)

    from sklearn.tree import DecisionTreeClassifier
    #min_samples_leaf - Quantidade mínima de amostras necessárias para ser uma folha
    #min_samples_split - Quantidade mínima de amostras necessárias para dividir um nó
    clf = DecisionTreeClassifier(random_state=0, min_samples_leaf= 20, min_samples_split=50)

    print("Iniciar treino....")

    model = clf.fit(treino_dados, treino_resultados)

    print("Treino concluido....")

    valor = model.score(teste_dados, teste_resultados)
    print( "Precisão por Árvore de Decisão: ", valor )


if __name__ == '__main__': # chamada da funcao principal
    main() # chamada da função main
