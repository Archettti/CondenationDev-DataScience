import streamlit as st
import pandas as pd
import altair as alt

def criarHistograma(coluna, df):
    histograma = alt.Chart(df, width=600).mark_bar().encode(
        alt.X(coluna, bin=True),
        y='count()', tooltip=[coluna, 'count()']
    ).interactive()
    return histograma


def criarBarras(coluna_num, coluna_cat, df):
    barras = alt.Chart(df, width=600).mark_bar().encode(
        x=alt.X(coluna_num, stack='zero'),
        y=alt.Y(coluna_cat),
        tooltip=[coluna_cat, coluna_num]
    ).interactive()
    return barras


def criarBoxplot(coluna_num, coluna_cat, df):
    boxplot = alt.Chart(df, width=600).mark_boxplot().encode(
        x=coluna_num,
        y=coluna_cat
    )
    return boxplot


def criarScatterplot(x, y, color, df):
    scatter = alt.Chart(df, width=800, height=400).mark_circle().encode(
        alt.X(x),
        alt.Y(y),
        color=color,
        tooltip=[x, y]
    ).interactive()
    return scatter


def criarCorrelacao(df, colunas_numericas):
    cor_data = (df[colunas_numericas]).corr().stack().reset_index().rename(
        columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'})
    cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
    base = alt.Chart(cor_data, width=800, height=800).encode(x='variable2:O', y='variable:O')
    text = base.mark_text().encode(text='correlation_label',
                                   color=alt.condition(alt.datum.correlation > 0.5, alt.value('white'),
                                                       alt.value('black')))

    cor_plot = base.mark_rect().encode(
        color='correlation:Q')

    return cor_plot + text


def calculaPontuacao(df, metrica):
    pontuacao = 10.0
    if metrica == 'Missing_Value':
        percentual = (df.isna().sum() / df.shape[0]) * 100  
        percentual = percentual.sum() / df.shape[1]
        
        if percentual > 50.0: pontuacao -=  5.0
        if percentual > 35.0 and percentual <= 50.0: pontuacao -=  4.0
        if percentual > 20.0 and percentual <= 35.0: pontuacao -=  3.0
        if percentual > 5.0 and percentual <= 20.0: pontuacao -=  2.0
        if percentual > 0.0 and percentual <= 5.0: pontuacao -=  1.0
    
    if metrica == 'Outliers':
        aux = pd.DataFrame({"colunas": df.columns, 'tipos': df.dtypes})
        colunas_numericas = list(aux[aux['tipos'] != 'object']['colunas'])
        
        q1 = df[colunas_numericas].quantile(0.25)
        q3 = df[colunas_numericas].quantile(0.75)
        iqr = q3-q1
        outliers_abaixo = df[colunas_numericas][df[colunas_numericas].apply(pd.to_numeric) < (q1 - 1.5*iqr)].count()
        outliers_acima = df[colunas_numericas][df[colunas_numericas].apply(pd.to_numeric) > (q3 + 1.5*iqr)].count()        
        outliers_soma = outliers_abaixo + outliers_acima
               
        percentual = (outliers_soma / df[colunas_numericas].shape[0]) * 100 
        percentual = percentual.sum() / df[colunas_numericas].shape[1]        

        if percentual > 50.0: pontuacao -=  5.0        
        if percentual > 35.0 and percentual <= 50.0: pontuacao -= 4.0
        if percentual > 20.0 and percentual <= 35.0: pontuacao -= 3.0
        if percentual > 5.0 and percentual <= 20.0: pontuacao -= 2.0
        if percentual > 0.0 and percentual <= 5.0: pontuacao -= 1.0
        
    return [percentual, pontuacao]
 
def main():
    
    st.title('Petrobras Conexões para a Inovação')
    opcoes = ('Dataset', 'Análise exploratória', 'Visualização dos dados', 'Métricas')  
    st.image('logo.png', width=300)
    st.subheader('Desafio Análise dos Dados para Aplicação de ML')
    
    st.sidebar.title('Opções')
    sidebarCategoria = st.sidebar.radio('', opcoes)
    
    file = st.file_uploader('Upload da base de dados para análise (.csv)', type='csv')
    
    if file is not None:
        index = st.checkbox('Utilizar primeira coluna como index')
        df = pd.read_csv(file, index_col=0) if index else pd.read_csv(file)

        aux = pd.DataFrame({"colunas": df.columns, 'tipos': df.dtypes})
        colunas_numericas = list(aux[aux['tipos'] != 'object']['colunas'])
        colunas_object = list(aux[aux['tipos'] == 'object']['colunas'])
        colunas = list(df.columns)
        
        st.subheader(sidebarCategoria) 
        
        # Dataset
        if (sidebarCategoria == opcoes[0]): 
            st.write('Número de observações: ', df.shape[0], 'Número de variáveis: ', df.shape[1])                      
                        
            st.markdown('**Dataset (describe)**')            
            st.table(df[colunas_numericas].describe().transpose())        
            
            st.markdown('**Dataset (overview)**')
            df_pivot = pd.DataFrame({'types': df.dtypes,
                                'nulls': df.isna().sum(),
                                '% nulls': df.isna().sum() / df.shape[0],
                                'size': df.shape[0],
                                'uniques': df.nunique()})
            st.table(df_pivot)

            
            numero_linhas = st.slider('Escolha o numero de linhas que deseja visualizar: ', min_value=5, max_value=50)
            st.dataframe(df.head(numero_linhas))

            st.markdown('**Observações (valores únicos)**')
            variavel = st.selectbox('Escolha uma variável: ', df.columns)
            st.dataframe(pd.DataFrame({variavel: df[variavel].unique()}))
        
        
        # Análise exploratória
        elif (sidebarCategoria == opcoes[1]):
            st.markdown('**Estimativas de Localização e Variabilidade**')
            
            col = st.selectbox('Selecione a coluna :', colunas_numericas)            
            if col is not None:
                st.markdown('Selecione a estimativa que deseja analisar :')
                
                mean = st.checkbox('Média')
                if mean:
                    st.markdown(df[col].mean())
                
                median = st.checkbox('Mediana')
                if median:
                    st.markdown(df[col].median())                    
                   
                desvio_pad = st.checkbox('Desvio padrão')
                if desvio_pad:
                    st.markdown(df[col].std())
                    
                amplitude_iqr = st.checkbox('Amplitude Interquatílica')
                if amplitude_iqr:
                    st.markdown(df[col].quantile(0.75) - df[col].quantile(0.25))

                st.markdown('**Quantidade de dados faltantes:**')
                dados_faltante = pd.DataFrame({'types': df.dtypes,
                                               'NA': df.isna().sum(),
                                               '% NA': (df.isna().sum() / df.shape[0]) * 100})
                st.table(dados_faltante)                

                st.markdown('**Quantidade de outliers:**')
                st.markdown('$$x ∉ [Q1 - 1.5 * {IQR}, Q3 + 1.5 * {IQR}] \Rightarrow x  { é outlier}$$')
                
                q1 = df[colunas_numericas].quantile(0.25)
                q3 = df[colunas_numericas].quantile(0.75)
                iqr = q3-q1
                outliers_abaixo = df[colunas_numericas][df[colunas_numericas].apply(pd.to_numeric) < (q1 - 1.5*iqr)].count()
                outliers_acima = df[colunas_numericas][df[colunas_numericas].apply(pd.to_numeric) > (q3 + 1.5*iqr)].count()
                  
                outliers = pd.DataFrame({'types': df[colunas_numericas].dtypes,
                                         'outliers abaixo': outliers_abaixo,
                                         'outliers acima': outliers_acima,
                                         '% outliers': ((outliers_abaixo + outliers_acima) / df[colunas_numericas].shape[0]) * 100})
                st.table(outliers)
                

        # Visualização dos dados       
        elif (sidebarCategoria == opcoes[2]):
           
            histograma = st.checkbox('Histograma')
            if histograma:
                col_num = st.selectbox('Selecione a Coluna Numerica: ', colunas_numericas, key='unique')
                st.markdown('Histograma da coluna' + str(col_num) + ': ')
                st.write(criarHistograma(col_num, df))
           
            barras = st.checkbox('Gráfico de barras')
            if barras:
                col_num_barras = st.selectbox('Selecione a coluna numerica: ', colunas_numericas, key='unique')
                col_cat_barras = st.selectbox('Selecione uma coluna categorica : ', colunas_object, key='unique')
                st.markdown('Gráfico de barras da coluna ' + str(col_cat_barras) + ' pela coluna ' + col_num_barras)
                st.write(criarBarras(col_num_barras, col_cat_barras, df))
            
            boxplot = st.checkbox('Boxplot')
            if boxplot:
                col_num_box = st.selectbox('Selecione a Coluna Numerica:', colunas_numericas, key='unique')
                col_cat_box = st.selectbox('Selecione uma coluna categorica : ', colunas_object, key='unique')
                st.markdown('Boxplot ' + str(col_cat_box) + ' pela coluna ' + col_num_box)
                st.write(criarBoxplot(col_num_box, col_cat_box, df))
           
            scatter = st.checkbox('Scatterplot')
            if scatter:
                col_num_x = st.selectbox('Selecione o valor de x ', colunas_numericas, key='unique')
                col_num_y = st.selectbox('Selecione o valor de y ', colunas_numericas, key='unique')
                col_color = st.selectbox('Selecione a coluna para cor', colunas)
                st.markdown('Selecione os valores de x e y')
                st.write(criarScatterplot(col_num_x, col_num_y, col_color, df))
            
            correlacao = st.checkbox('Correlacao')
            if correlacao:
                st.markdown('Gráfico de correlação das colunas númericas')
                st.write(criarCorrelacao(df, colunas_numericas))
            
            
        # Metricas      
        elif (sidebarCategoria == opcoes[3]):
            metrica = ('Missing_Value', 'Outliers')
            
            missing_value = st.checkbox('Missing_value')
            if missing_value:
                percentual, pontuacao_mv = calculaPontuacao(df, metrica[0])
                st.markdown('Média dos dados faltantes do dataset acima de 50%: -5 pontos')
                st.markdown('Média dos dados faltantes do dataset entre 36 e 50%: -4 pontos')
                st.markdown('Média dos dados faltantes do dataset entre 21 e 35%: -3 pontos')
                st.markdown('Média dos dados faltantes do dataset entre 6 e 20%: -2 pontos')
                st.markdown('Média dos dados faltantes do dataset entre 1 e 5%: -1 ponto')
                            
                st.write('Percentual: ', round(percentual,3), 'Pontuação Dataset Dados Faltantes: ', pontuacao_mv)
              
            outliers = st.checkbox('Outliers')
            if outliers:
                percentual, pontuacao_out = calculaPontuacao(df, metrica[1])
                st.markdown('Média dos outliers do dataset acima de 50%: -5 pontos')
                st.markdown('Média dos outliers do dataset entre 36 e 50%: -4 pontos')
                st.markdown('Média dos outliers do dataset entre 21 e 35%: -3 pontos')
                st.markdown('Média dos outliers do dataset entre 6 e 20%: -2 pontos')
                st.markdown('Média dos outliers do dataset entre 1 e 5%: -1 ponto')
                
                st.write('Percentual: ', round(percentual,3), 'Pontuação Dataset Outliers: ', pontuacao_out)
            
            st.write('**Pontuação Média Geral do Dataset: **', (calculaPontuacao(df, metrica[0])[1] + calculaPontuacao(df, metrica[1])[1]) / 2)
        
if __name__ == '__main__':
    main()


    