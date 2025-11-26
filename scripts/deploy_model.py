#!/usr/bin/env python3
"""
Script de Deploy - Demonstração de Uso do Modelo Treinado
Sistema de Predição de Evasão Estudantil
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_model_and_preprocessors():
    """Carrega o modelo e os pré-processadores salvos"""
    base_path = Path(__file__).parent.parent
    
    # Carregar modelo
    model_path = base_path / 'modelo_final.pkl'
    
    if not model_path.exists():
        print("❌ ERRO: Modelo não encontrado!")
        print(f"   O arquivo {model_path} não existe.")
        print("\n📝 Para resolver este problema:")
        print("   1. Abra o Jupyter Notebook: jupyter notebook")
        print("   2. Execute o notebook: notebooks/02_modelagem_avaliacao.ipynb")
        print("   3. Execute todas as células do notebook")
        print("   4. O modelo será salvo automaticamente como 'modelo_final.pkl'")
        print("\n   Depois disso, você poderá executar este script novamente.")
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Modelo carregado de: {model_path}")
    
    # Carregar scaler (se existir)
    scaler_path = base_path / 'scaler.pkl'
    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler carregado de: {scaler_path}")
    
    # Carregar label encoders (se existir)
    encoders_path = base_path / 'label_encoders.pkl'
    label_encoders = {}
    if encoders_path.exists():
        label_encoders = joblib.load(encoders_path)
        print(f"✅ Label encoders carregados de: {encoders_path}")
    
    return model, scaler, label_encoders

def prepare_new_student_data(student_data, label_encoders):
    """
    Prepara os dados de um novo estudante para predição
    
    Args:
        student_data: Dicionário com os dados do estudante
        label_encoders: Dicionário com os encoders para variáveis categóricas
        
    Returns:
        DataFrame preparado
    """
    # Criar DataFrame
    df = pd.DataFrame([student_data])
    
    # Aplicar label encoding nas variáveis categóricas
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    return df

def predict_dropout_risk(model, student_data, scaler=None, label_encoders=None):
    """
    Faz predição de risco de evasão para um novo estudante
    
    Args:
        model: Modelo treinado
        student_data: Dicionário com dados do estudante
        scaler: Scaler para normalização (opcional)
        label_encoders: Encoders para variáveis categóricas (opcional)
        
    Returns:
        Tupla (predição, probabilidade)
    """
    # Preparar dados
    if label_encoders:
        df = prepare_new_student_data(student_data, label_encoders)
    else:
        df = pd.DataFrame([student_data])
    
    # Remover student_id se existir
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
    
    # Normalizar se necessário
    if scaler:
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
    else:
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
    
    return prediction, probability

def main():
    """Função principal - Demonstração de uso do modelo"""
    print("=" * 70)
    print("SISTEMA DE PREDIÇÃO DE EVASÃO ESTUDANTIL - DEPLOY")
    print("=" * 70)
    print()
    
    # Carregar modelo e pré-processadores
    print("📦 Carregando modelo e pré-processadores...")
    model, scaler, label_encoders = load_model_and_preprocessors()
    print()
    
    # Exemplo 1: Estudante com alto risco de evasão
    print("=" * 70)
    print("EXEMPLO 1: Estudante com ALTO RISCO de Evasão")
    print("=" * 70)
    
    student_high_risk = {
        'age': 20,
        'gender': 'M',
        'socioeconomic_level': 2,
        'avg_grade': 4.5,
        'avg_attendance': 60.0,
        'current_semester': 1,
        'total_enrollments': 4,
        'failed_courses': 2,
        'completed_courses': 1,
        'total_interactions': 10,
        'unique_sessions_count': 3,
        'total_duration_hours': 5.0,
        'days_since_last_interaction': 15,
        'engagement_score': 20.0,
        'scholarship_percentage': 0,
        'overdue_payments': 3,
        'pending_payments': 2,
        'outstanding_amount': 5000.0,
        'success_rate': 25.0,
        'failure_rate': 50.0,
        'interaction_per_enrollment': 2.5
    }
    
    print("\n📋 Dados do Estudante:")
    for key, value in student_high_risk.items():
        print(f"   - {key}: {value}")
    
    prediction, probability = predict_dropout_risk(
        model, student_high_risk, scaler, label_encoders
    )
    
    print(f"\n🎯 RESULTADO DA PREDIÇÃO:")
    print(f"   - Risco de Evasão: {'SIM' if prediction == 1 else 'NÃO'}")
    print(f"   - Probabilidade de Evasão: {probability[1]*100:.2f}%")
    print(f"   - Probabilidade de Permanência: {probability[0]*100:.2f}%")
    
    if prediction == 1:
        print(f"\n⚠️  ALERTA: Este estudante apresenta ALTO RISCO de evasão!")
        print(f"   Recomenda-se intervenção imediata.")
    else:
        print(f"\n✅ Este estudante apresenta BAIXO RISCO de evasão.")
    
    print("\n" + "=" * 70)
    
    # Exemplo 2: Estudante com baixo risco de evasão
    print("EXEMPLO 2: Estudante com BAIXO RISCO de Evasão")
    print("=" * 70)
    
    student_low_risk = {
        'age': 22,
        'gender': 'F',
        'socioeconomic_level': 4,
        'avg_grade': 8.5,
        'avg_attendance': 95.0,
        'current_semester': 3,
        'total_enrollments': 12,
        'failed_courses': 0,
        'completed_courses': 11,
        'total_interactions': 80,
        'unique_sessions_count': 25,
        'total_duration_hours': 120.0,
        'days_since_last_interaction': 2,
        'engagement_score': 150.0,
        'scholarship_percentage': 50,
        'overdue_payments': 0,
        'pending_payments': 0,
        'outstanding_amount': 0.0,
        'success_rate': 91.67,
        'failure_rate': 0.0,
        'interaction_per_enrollment': 6.67
    }
    
    print("\n📋 Dados do Estudante:")
    for key, value in student_low_risk.items():
        print(f"   - {key}: {value}")
    
    prediction, probability = predict_dropout_risk(
        model, student_low_risk, scaler, label_encoders
    )
    
    print(f"\n🎯 RESULTADO DA PREDIÇÃO:")
    print(f"   - Risco de Evasão: {'SIM' if prediction == 1 else 'NÃO'}")
    print(f"   - Probabilidade de Evasão: {probability[1]*100:.2f}%")
    print(f"   - Probabilidade de Permanência: {probability[0]*100:.2f}%")
    
    if prediction == 1:
        print(f"\n⚠️  ALERTA: Este estudante apresenta ALTO RISCO de evasão!")
        print(f"   Recomenda-se intervenção imediata.")
    else:
        print(f"\n✅ Este estudante apresenta BAIXO RISCO de evasão.")
    
    print("\n" + "=" * 70)
    print("✅ Demonstração concluída!")
    print("=" * 70)

if __name__ == "__main__":
    main()

