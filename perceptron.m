function [w,b] = perceptron(o, w, max_it, taxaAprendizado, x, d, xValidacao, dValidacao)
%{
FUNÇÃO: Rede Perceptron que reconhece três espécies de flores iris.
        Características da rede: Perceptron simples de uma única camada. É
        composto por 4 entradas (neurônios), um para cada atributo 
        (comprimento e largura da sepala e comprimento e largura da pétala)
        Possui 3 saídas possíveis, uma para cada classe
ENTRADA:
    o: número inteiro       -> Número de saídas
    w: matriz real          -> Matriz oxm de pesos da rede
    max_it: número inteiro  -> Número máximo de iterações
    alfa: número real       -> Taxa de aprendizagem
    dados: matriz real      -> Conjunto de treinamento
    validacao: matriz real  -> Conjunto de validação
SAÍDA:
    w: matriz real de pesos (oxm)
    b: vetor real de bias
%}
    % --------------- Parâmetros gerais para o treinamento ---------------- 
   
    n = size(x,1);              % Número de amostras
    
    % d é a matriz de saidas esperadas contendo o valor 1 na n-ésima coluna
    % tal que n é a classe a qual ela pertence. Cada linha refere-se a um 
    % padrão de entrada. Matriz nxo.
    d = saidaEsperada(d,o);  
    
    % ---------------- Parâmetros gerais para a validação -----------------
    
    dValidacao = saidaEsperada(dValidacao,o);
    k = size(xValidacao,1);     % Número de amostras para validação
    
    % ---------------------------------------------------------------------
    
    % Inicializando b como zero
    b = zeros(o,1);     % Um bias para cada saída
    
    t = 1;              % Iteração          
    E = 1;              % Erro quadrático
    
    while(t < max_it && E > 0)    
        E = 0;
        EValidacao = 0;
        for i = 1 : n                   % Para cada padrão de treinamento
                                                
            y = softmax(w*x(i,:)'+ b);  % Saída da rede para xi. A entrada  
                                        % da função é um vetor de tamanho 
                                        % ox1. y tem tamanho ox1
                                                
            e = d(i,:)' - y;            % Determinar o erro para xi
            w = w + taxaAprendizado * e * x(i,:);  % Atualizar vetor de pesos
            b = b + taxaAprendizado * e;           % Atualizar o termo de bias
            E = E + sum(e.^2);          % Acumular o erro
        end 
        emq(t) = E / n;               % Erro quadrático médio da iteração
        
        % ---------------------- VALIDAÇÃO DA ÉPOCA -----------------------
        
        for i = 1 : k                   % Para cada padrão de validação
            y = softmax(w*xValidacao(i,:)'+ b);
            e = dValidacao(i,:)' - y;
            EValidacao = EValidacao + sum(e.^2);
        end
        
            emqValidacao(t) = EValidacao / k;
            
        if(t ~= 1)
            % Se o erro quadrático médio da validação aumenta, o
            % treinamento da rede é interrompido
            if(emqValidacao(t) > emqValidacao(t-1))
                break;
            end
        end
        
         % ----------------------------------------------------------------
        
         t = t + 1;
    end
    
    % ------------- Plotar Gráfico de Erro Médio Quadrático ---------------
%     figure
%     x = 1: size(emq,2);
%     plot(x,emq,x,emqValidacao,'--');
%     title('Gráfico de Erro Médio Quadrático');
%     xlabel('Iterações - t');
%     ylabel('Erro médio quadrático');
%     legend('Treinamento','Validação');
    
    % ---------------------------------------------------------------------
       
end

function [y] = saidaEsperada(d,o)
%{
ENTRADA:
    d: matriz de inteiros nx1 (n padrões e 1 coluna) -> saída esperada da 
    rede. A qual classe pertence o n-ésimo padrão.
    o : número inteiro -> número de saídas da rede
SAÍDA:
    y: matriz binária n x o (n padrões, o saídas) -> retorna 1 na posição do 
    padrão que pertence e 0 no restante. Por exemplo, se o padrão pertence 
    a classe 3, sua saída esperada é y(i) = [ 0 0 1 ... 0]
%}
    n = size(d,1);
    y = zeros(n,o);
    
    for i = 1:n
        y(i,d(i)) = 1;
    end
end