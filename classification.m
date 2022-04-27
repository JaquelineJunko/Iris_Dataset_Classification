% Rede Perceptron que reconhece três espécies de flores iris. 
% Características da rede: Perceptron simples de uma única camada. É 
% composto por 4 entradas (neurônios), um para cada atributo (comprimento e 
% largura da sepala e comprimento e largura da pétala). Possui 3 saídas 
% possíveis, uma para cada classe.
        
% --------------------- FORMATO DOS DADOS ÍRIS ---------------------
% |                       |                       | 1 - Setosa     |
% |        Sépala         |         Pétala        | 2 - Virginica  |
% |                       |                       | 3 - Versicolor |
% | Comprimento | Largura | Comprimento | Largura |     Classe     |
% ------------------------------ ÍRIS ------------------------------

n = 150;                    % Número de amostras
m = 5;                      % Número de atributos + Classe que pertence
o = 3;                      % Número de saídas

arquivo = fopen("data.txt");
dados = fscanf(arquivo,"%f,%f,%f,%f,%d",[m,n]);
fclose(arquivo);
dados = dados';             % cada linha apresenta os padrões de uma flor

% zscore - normalizar dados exceto a última coluna, pois refere-se a qual classe a flor pertence.
dados(:,1:m-1) = zscore(dados(:,1:m-1));


% ---------- SEPARAR CONJUNTO DE TREINAMENTO, TESTE E VALIDAÇÃO -----------

% -------------------------------- AMOSTRA --------------------------------
% |            TREINO               |       TESTE       |     VALIDAÇÃO   |
% -------------------------------------------------------------------------

% treino - número aleatório entre 70% do conjunto                                 
treino = 105;                                 
% Conjunto de validação e teste serão sempre a metade restante da amostra

% teste 15% do conjunto
teste = 23;

% embaralhar dados
dados = dados(randperm(size(dados,1)),:);

% -------------------- TREINAMENTO E VALIDAÇÃO DA REDE --------------------

x = dados(1:treino,1:m-1);                      % Entradas do treinamento
d = dados(1:treino,m);                          % Saída esperada 
xValidacao = dados(treino+teste+1 : n, 1:m-1);  % Entradas para validar
dValidacao = dados(treino+teste+1 : n, m);      % Saída esperada


%   Inicializando a matriz de Pesos aleatória
% w1 = rand(o,m-1);

w = zeros(o,m-1);
max_it = 300;
taxaAprendizado = 0.3;

[w,bias] = perceptron(o, w, max_it, taxaAprendizado, x, d, xValidacao, dValidacao);

% Matriz de Confusão do Treinamento
mcTreinamento = matrizConfusao(w, bias, x, d);

% Matriz de Confusão da Validação
mcValidacao = matrizConfusao(w, bias, xValidacao, dValidacao);


% --------------------------- TESTE DA REDE -------------------------------
% Gerar Matriz de Confusão do Teste
mcTeste = matrizConfusao(w, bias, dados(treino+1 : treino+teste, 1:m-1), dados(treino+1 : treino+teste, m));

acuracia = trace(mcTeste)/sum(sum(mcTeste));
% -------------------------------------------------------------------------


% ---------------------- Exibir dados do Experimento ----------------------

% Acurácia
fprintf("ACURÁCIA da rede: %f\n\n", acuracia);

% Taxa de Aprendizado
fprintf("Taxa de Aprendizado: %f\n\n", taxaAprendizado);

%Número de Iterações Maxima
fprintf("Número de Iterações Maxima: %d\n\n", max_it);

% w após treinamento
fprintf("PESOS após treinamento da rede\n");
disp(w);

% Bias
fprintf("Bias\n")
disp(bias);

% Número de amostras para cada fase
fprintf("Número de amostras no TREINAMENTO: %d\n", treino);
fprintf("Número de amostras na VALIDAÇÃO: %d\n", n-treino-teste);
fprintf("Número de amostras no TESTE: %d\n\n", teste);

% Exibir matrizes de confusão
fprintf("Matriz de Confusão do TREINAMENTO\n");
disp(mcTreinamento);
fprintf("Acertos: %d\nErros: %d\n\n", trace(mcTreinamento), treino - trace(mcTreinamento) );

fprintf("Matriz de Confusão da VALIDAÇÃO\n");
disp(mcValidacao);
fprintf("Acertos: %d\nErros: %d\n\n", trace(mcValidacao), n-treino-teste - trace(mcValidacao));

fprintf("Matriz de Confusão do TESTE\n");
disp(mcTeste);
fprintf("Acertos: %d\nErros: %d\n\n", trace(mcTeste), teste - trace(mcTeste) );

%%%%%%%%%%%%% Dados para Experimento %%%%%%%%%%%%%%%%%%%%%%%%%

w1 = rand(o,m-1);

max_it = [10 ;50; 100; 300; 500];
taxaAprendizado = [0.01; 0.05; 0.1; 0.3; 0.5; 1];

for j = 1: size(max_it,1) % para cada num de max_it
    
    for k = 1: size(taxaAprendizado,1)
        acuracia = 0;
        
        for i = 1: 10
            
            % embaralhar dados
            dados = dados(randperm(size(dados,1)),:);

            % -------------------- TREINAMENTO E VALIDAÇÃO DA REDE --------------------

            x = dados(1:treino,1:m-1);                      % Entradas do treinamento
            d = dados(1:treino,m);                          % Saída esperada 
            xValidacao = dados(treino+teste+1 : n, 1:m-1);  % Entradas para validar
            dValidacao = dados(treino+teste+1 : n, m);      % Saída esperada
            
            w = w1;

            [w,bias] = perceptron(o, w, max_it(j), taxaAprendizado(k), x, d, xValidacao, dValidacao);

            mcTeste = matrizConfusao(w, bias, dados(treino+1 : treino+teste, 1:m-1), dados(treino+1 : treino+teste, m));

            acuracia = acuracia + trace(mcTeste)/sum(sum(mcTeste));            
        end
         aleatorio(k,j) = acuracia/10;
    end   
end
    
%%%%%%%%%%%%% Dados para Experimento %%%%%%%%%%%%%%%%%%%%%%%%%

figure;
plot(max_it,aleatorio(1,:),'y',max_it,aleatorio(2,:),'m',max_it,aleatorio(3,:),'c',max_it,aleatorio(4,:),'r',max_it,aleatorio(5,:),'g',max_it,aleatorio(6,:),'b');
title('Média das Acurácias Médias - w inicial aletório');
xlabel('Iterações máxima');
ylabel('Acurácia');
legend("0.01", "0.05", "0.1", "0.3", "0.5", "1",'Location','southeast' );
title(legend,'Taxas de Aprendizado')