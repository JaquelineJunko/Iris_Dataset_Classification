function [mc] = matrizConfusao(w,bias,x,d)
    %{
    ENTRADA:
        w: matriz real oxm de pesos. 'o' é o número de saídas do perceptron e 'm' é o número de neurônios da rede
        bias: vetor real ox1                                    
        dados: matriz real contendo os 'o' dados das n amostras
        classes: número inteiro. Quantidade de classes e também o número de saídas possíveis da rede.
        
    SAÍDA:
        mc: matriz inteira   -> matriz de confusão
    %}
        mc = zeros( size(w,1) );
        n = size(x,1);
            
        for k = 1: n
            y = softmax(w*x(k,:)'+ bias);
            i = d(k);           % classe esperada
            j = find(y == max(y));  % classe atribuida pela rede
            mc(i,j) = mc(i,j) + 1;
        end
    end