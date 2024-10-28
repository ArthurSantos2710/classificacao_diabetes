import streamlit as st
import pickle
import numpy as np

#Exercício: Montar uma interface no Streamlit para classificar uma pessoa como portadora de diabetes ou não.
#O modelo de machine learning já está treinado e salvo em um arquivo chamado "trained_model.sav".

#os dados de entrada são:
#1. Número de vezes grávida
#2. Concentração de glicose
#3. Pressão sanguínea
#4. Espessura da pele
#5. Insulina
#6. IMC
#7. Função de pedigree de diabetes
#8. Idade

#todos esses dados são numéricos

#o input do modelo deve ser um array numpy 2d com todas features listadas acima nessa ordem

#o modelo deve retornar 0 ou 1
#se o resultado for 1, a pessoa é portadora de diabetes
#se o resultado for 0, a pessoa não é portadora de diabetes


def load_model():
    model = pickle.load(open('trained_model.sav', 'rb'))
    return model

def main():
    st.title("Classificação de pessoas diabéticas")
    
    gravidez = st.slider(label = "Número de vezes grávida", min_value = 0, max_value = 20, step = 1)
    glicose = st.number_input("Concentração de glicose no sangue", min_value = 0)
    pressao = st.number_input("Pressão sanguínea", min_value = 0)
    pele = st.number_input("Espessura da pele", min_value = 0)
    insulina = st.number_input("Quantidade de insulina presente no sangue", min_value=0)
    imc = st.slider("Índice de Massa Corporal (IMC)", min_value = 18.5, max_value = 60.0, format="%.1f")
    funcao_pedigree = st.number_input("Função de pedigree de diabetes", min_value=0.0, format="%.2f")
    idade = st.number_input("Idade", min_value=0, step=1)

    if st.button("Realizar classificação"):
        modelo = load_model()
        entradas = np.array([[gravidez, glicose, pressao, pele, insulina, imc, funcao_pedigree, idade]])
        resultado = modelo.predict(entradas)

        if resultado[0] == 1:
            st.write("A pessoa porta diabetes")
        else:
            st.write("A pessoa não porta diabetes")

if __name__ == '__main__':
    main()