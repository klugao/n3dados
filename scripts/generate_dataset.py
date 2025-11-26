#!/usr/bin/env python3
"""
Script para gerar dataset sintético de predição de evasão estudantil
Baseado no schema do projeto N1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_student_dropout_dataset(n_students=1000, random_seed=42):
    """
    Gera dataset sintético para predição de evasão estudantil
    
    Args:
        n_students: Número de estudantes a gerar
        random_seed: Seed para reprodutibilidade
        
    Returns:
        DataFrame com features e target
    """
    np.random.seed(random_seed)
    
    # IDs dos estudantes
    student_ids = [f'STU{i:04d}' for i in range(1, n_students + 1)]
    
    # Features Demográficas
    age = np.random.normal(22, 3, n_students).astype(int)
    age = np.clip(age, 18, 35)  # Limitar entre 18 e 35 anos
    
    gender = np.random.choice(['M', 'F', 'O'], n_students, p=[0.5, 0.48, 0.02])
    socioeconomic_level = np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    
    # Features Acadêmicas
    avg_grade = np.random.normal(6.5, 1.5, n_students)
    avg_grade = np.clip(avg_grade, 0, 10)
    
    avg_attendance = np.random.normal(80, 15, n_students)
    avg_attendance = np.clip(avg_attendance, 0, 100)
    
    current_semester = np.random.choice([1, 2, 3, 4, 5, 6], n_students, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
    total_enrollments = current_semester * 4 + np.random.randint(-2, 3, n_students)
    total_enrollments = np.clip(total_enrollments, 1, 30)
    
    failed_courses = np.random.poisson(0.5, n_students)
    failed_courses = np.clip(failed_courses, 0, 5)
    
    completed_courses = total_enrollments - failed_courses - np.random.randint(0, 3, n_students)
    completed_courses = np.clip(completed_courses, 0, total_enrollments)
    
    # Features Comportamentais
    total_interactions = np.random.poisson(50, n_students)
    total_interactions = np.clip(total_interactions, 0, 200)
    
    unique_sessions_count = np.random.poisson(15, n_students)
    unique_sessions_count = np.clip(unique_sessions_count, 0, 50)
    
    total_duration_hours = np.random.gamma(2, 10, n_students)
    total_duration_hours = np.clip(total_duration_hours, 0, 200)
    
    days_since_last_interaction = np.random.exponential(5, n_students)
    days_since_last_interaction = np.clip(days_since_last_interaction, 0, 30)
    
    engagement_score = (total_interactions * 0.3 + 
                       unique_sessions_count * 2 + 
                       total_duration_hours * 0.5)
    
    # Features Financeiras
    scholarship_percentage = np.random.choice([0, 25, 50, 75, 100], n_students, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    overdue_payments = np.random.poisson(0.3, n_students)
    overdue_payments = np.clip(overdue_payments, 0, 5)
    
    pending_payments = np.random.poisson(0.5, n_students)
    pending_payments = np.clip(pending_payments, 0, 5)
    
    outstanding_amount = (overdue_payments + pending_payments) * np.random.uniform(500, 2000, n_students)
    
    # Criar DataFrame
    df = pd.DataFrame({
        'student_id': student_ids,
        # Demográficas
        'age': age,
        'gender': gender,
        'socioeconomic_level': socioeconomic_level,
        # Acadêmicas
        'avg_grade': avg_grade,
        'avg_attendance': avg_attendance,
        'current_semester': current_semester,
        'total_enrollments': total_enrollments,
        'failed_courses': failed_courses,
        'completed_courses': completed_courses,
        # Comportamentais
        'total_interactions': total_interactions,
        'unique_sessions_count': unique_sessions_count,
        'total_duration_hours': total_duration_hours,
        'days_since_last_interaction': days_since_last_interaction,
        'engagement_score': engagement_score,
        # Financeiras
        'scholarship_percentage': scholarship_percentage,
        'overdue_payments': overdue_payments,
        'pending_payments': pending_payments,
        'outstanding_amount': outstanding_amount
    })
    
    # Calcular probabilidade de evasão baseada nas features
    # Fatores que aumentam risco de evasão:
    risk_score = 0.0
    
    # Fator acadêmico (peso 40%)
    risk_score += np.where(df['avg_grade'] < 6.0, (6.0 - df['avg_grade']) / 6.0 * 40, 0)
    risk_score += np.where(df['avg_attendance'] < 75.0, (75.0 - df['avg_attendance']) / 75.0 * 30, 0)
    risk_score += df['failed_courses'] * 5
    
    # Fator financeiro (peso 25%)
    risk_score += df['overdue_payments'] * 8
    risk_score += np.minimum(df['outstanding_amount'] / 1000, 15)
    risk_score += np.where(df['scholarship_percentage'] == 0, 5, 0)
    
    # Fator comportamental (peso 25%)
    risk_score += np.where(df['total_interactions'] < 20, 10, 0)
    risk_score += np.where(df['days_since_last_interaction'] > 7, 8, 0)
    risk_score += np.where(df['engagement_score'] < 50, 7, 0)
    
    # Fator demográfico (peso 10%)
    risk_score += np.where(df['socioeconomic_level'] <= 2, 5, 0)
    risk_score += np.where(df['current_semester'] == 1, 3, 0)  # Primeiro semestre mais vulnerável
    
    # Normalizar para probabilidade (0-1)
    risk_probability = 1 / (1 + np.exp(-(risk_score - 50) / 15))  # Sigmoid
    
    # Gerar target binário baseado na probabilidade
    dropout = np.random.binomial(1, risk_probability, n_students)
    
    df['dropout'] = dropout
    
    # Adicionar algumas features derivadas
    df['success_rate'] = (df['completed_courses'] / df['total_enrollments'].replace(0, np.nan)) * 100
    df['failure_rate'] = (df['failed_courses'] / df['total_enrollments'].replace(0, np.nan)) * 100
    df['interaction_per_enrollment'] = df['total_interactions'] / df['total_enrollments'].replace(0, np.nan)
    
    # Preencher valores NaN com 0
    df = df.fillna(0)
    
    return df

def main():
    """Função principal para gerar e salvar o dataset"""
    print("🔄 Gerando dataset sintético de predição de evasão estudantil...")
    
    # Gerar dataset
    df = generate_student_dropout_dataset(n_students=1000, random_seed=42)
    
    # Criar diretório data se não existir
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Salvar dataset
    output_path = data_dir / 'student_dropout_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset gerado com sucesso!")
    print(f"   📁 Localização: {output_path}")
    print(f"   📊 Total de registros: {len(df)}")
    print(f"   📈 Features: {len(df.columns) - 1}")  # -1 para excluir target
    print(f"   🎯 Taxa de evasão: {df['dropout'].mean()*100:.1f}%")
    print(f"   📋 Colunas: {', '.join(df.columns.tolist())}")

if __name__ == "__main__":
    main()

