import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class StudentPerformanceAnalytics:
    """
    Advanced student performance analytics system for:
    - At-risk student identification
    - Learning pattern clustering
    - Performance trend analysis
    - Personalized improvement recommendations
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.students_data = {}
        self.analytics_results = {}
        self.risk_thresholds = {
            'critical': 0.3,
            'high': 0.5,
            'medium': 0.7,
            'low': 0.85
        }
        
        # Load and process data
        self.load_student_data()
        self.process_analytics()
    
    def load_student_data(self):
        """Load and consolidate all student data"""
        try:
            # Load CSV files
            activity_path = self.data_dir / "student_activity.csv"
            scores_path = self.data_dir / "student_scores.csv" 
            content_path = self.data_dir / "student_content.csv"
            
            self.activity_df = pd.read_csv(activity_path) if activity_path.exists() else pd.DataFrame()
            self.scores_df = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
            self.content_df = pd.read_csv(content_path) if content_path.exists() else pd.DataFrame()
            
            # Normalize student IDs
            if not self.activity_df.empty:
                self.activity_df['student_id'] = self.activity_df['student_id'].str.lower()
            if not self.scores_df.empty:
                self.scores_df['student_id'] = self.scores_df['student_id'].str.lower()
                self.scores_df['topic'] = self.scores_df['topic'].str.lower()
            if not self.content_df.empty:
                self.content_df['topic'] = self.content_df['topic'].str.lower()
            
            print(f"Loaded data: Activity({len(self.activity_df)}), Scores({len(self.scores_df)}), Content({len(self.content_df)})")
            
        except Exception as e:
            print(f"Error loading student data: {e}")
            self.activity_df = pd.DataFrame()
            self.scores_df = pd.DataFrame()
            self.content_df = pd.DataFrame()
    
    def process_analytics(self):
        """Process all analytics computations"""
        try:
            self.compute_student_metrics()
            self.identify_at_risk_students()
            self.cluster_learning_patterns()
            self.generate_performance_insights()
            self.create_improvement_recommendations()
            
        except Exception as e:
            print(f"Error processing analytics: {e}")
    
    def compute_student_metrics(self):
        """Compute comprehensive metrics for each student"""
        student_metrics = {}
        
        # Get all unique students
        students = set()
        if not self.activity_df.empty:
            students.update(self.activity_df['student_id'].unique())
        if not self.scores_df.empty:
            students.update(self.scores_df['student_id'].unique())
        
        for student_id in students:
            metrics = self._calculate_individual_metrics(student_id)
            student_metrics[student_id] = metrics
        
        self.student_metrics = student_metrics
        return student_metrics
    
    def _calculate_individual_metrics(self, student_id: str) -> Dict:
        """Calculate detailed metrics for a single student"""
        student_activity = self.activity_df[self.activity_df['student_id'] == student_id] if not self.activity_df.empty else pd.DataFrame()
        student_scores = self.scores_df[self.scores_df['student_id'] == student_id] if not self.scores_df.empty else pd.DataFrame()
        
        metrics = {
            'student_id': student_id,
            'total_time_spent': float(student_activity['time_spent'].sum() if not student_activity.empty else 0),
            'avg_score': float(student_scores['score'].mean() if not student_scores.empty else 0),
            'max_score_possible': float(student_scores['max_score'].sum() if not student_scores.empty else 0),
            'total_attempts': int(student_scores.shape[0] if not student_scores.empty else 0),
            'topics_attempted': int(student_scores['topic'].nunique() if not student_scores.empty else 0),
            'completion_rate': 0.0,
            'consistency_score': 0.0,
            'improvement_trend': 0.0,
            'engagement_level': 'low',
            'performance_category': 'struggling',
            'risk_level': 'high'
        }
        
        if not student_scores.empty:
            # Performance calculations
            scores = student_scores['score'].values
            max_scores = student_scores['max_score'].values
            
            # Completion rate (percentage of max possible score achieved)
            if max_scores.sum() > 0:
                metrics['completion_rate'] = float(scores.sum() / max_scores.sum())
            
            # Consistency score (inverse of coefficient of variation)
            if len(scores) > 1 and scores.mean() > 0:
                cv = scores.std() / scores.mean()
                metrics['consistency_score'] = float(max(0, 1 - cv))
            
            # Improvement trend (correlation with attempt order)
            if len(scores) > 2:
                x = np.arange(len(scores))
                correlation = np.corrcoef(x, scores)[0, 1]
                metrics['improvement_trend'] = float(correlation if not np.isnan(correlation) else 0)
            
            # Performance categorization
            avg_percentage = (scores.mean() / max_scores.mean()) * 100 if max_scores.mean() > 0 else 0
            if avg_percentage >= 85:
                metrics['performance_category'] = 'excellent'
            elif avg_percentage >= 70:
                metrics['performance_category'] = 'good'
            elif avg_percentage >= 50:
                metrics['performance_category'] = 'average'
            else:
                metrics['performance_category'] = 'struggling'
        
        # Engagement level based on activity
        if not student_activity.empty:
            total_time = student_activity['time_spent'].sum()
            if total_time >= 200:
                metrics['engagement_level'] = 'high'
            elif total_time >= 100:
                metrics['engagement_level'] = 'medium'
            else:
                metrics['engagement_level'] = 'low'
        
        # Risk assessment
        risk_score = self._calculate_risk_score(metrics)
        metrics['risk_score'] = risk_score
        
        if risk_score <= self.risk_thresholds['critical']:
            metrics['risk_level'] = 'critical'
        elif risk_score <= self.risk_thresholds['high']:
            metrics['risk_level'] = 'high'
        elif risk_score <= self.risk_thresholds['medium']:
            metrics['risk_level'] = 'medium'
        else:
            metrics['risk_level'] = 'low'
        
        return metrics
    
    def _calculate_risk_score(self, metrics: Dict) -> float:
        """Calculate overall risk score (0-1, lower is higher risk)"""
        weights = {
            'avg_score': 0.3,
            'completion_rate': 0.25,
            'consistency_score': 0.15,
            'engagement_level': 0.15,
            'improvement_trend': 0.15
        }
        
        # Normalize scores to 0-1 range
        score_norm = min(metrics['avg_score'] / 20, 1.0)  # Assuming max score of 20
        completion_norm = metrics['completion_rate']
        consistency_norm = metrics['consistency_score']
        
        engagement_norm = {'low': 0.2, 'medium': 0.6, 'high': 1.0}[metrics['engagement_level']]
        trend_norm = max(0, (metrics['improvement_trend'] + 1) / 2)  # Convert -1,1 to 0,1
        
        risk_score = (
            score_norm * weights['avg_score'] +
            completion_norm * weights['completion_rate'] +
            consistency_norm * weights['consistency_score'] +
            engagement_norm * weights['engagement_level'] +
            trend_norm * weights['improvement_trend']
        )
        
        return float(risk_score)
    
    def identify_at_risk_students(self) -> Dict:
        """Identify students at risk of poor performance"""
        at_risk_students = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for student_id, metrics in self.student_metrics.items():
            risk_level = metrics['risk_level']
            at_risk_students[risk_level].append({
                'student_id': student_id,
                'risk_score': metrics['risk_score'],
                'avg_score': metrics['avg_score'],
                'completion_rate': metrics['completion_rate'],
                'engagement_level': metrics['engagement_level'],
                'main_issues': self._identify_main_issues(metrics)
            })
        
        # Sort by risk score (ascending - most at risk first)
        for risk_level in at_risk_students:
            at_risk_students[risk_level].sort(key=lambda x: x['risk_score'])
        
        self.at_risk_students = at_risk_students
        return at_risk_students
    
    def _identify_main_issues(self, metrics: Dict) -> List[str]:
        """Identify main performance issues for a student"""
        issues = []
        
        if metrics['avg_score'] < 10:
            issues.append('Low test scores')
        if metrics['completion_rate'] < 0.5:
            issues.append('Poor completion rate')
        if metrics['consistency_score'] < 0.3:
            issues.append('Inconsistent performance')
        if metrics['engagement_level'] == 'low':
            issues.append('Low engagement')
        if metrics['improvement_trend'] < -0.2:
            issues.append('Declining performance')
        if metrics['topics_attempted'] < 2:
            issues.append('Limited topic coverage')
        
        return issues if issues else ['General performance concerns']
    
    def cluster_learning_patterns(self) -> Dict:
        """Cluster students based on learning patterns"""
        try:
            # Prepare feature matrix
            features = []
            student_ids = []
            
            for student_id, metrics in self.student_metrics.items():
                feature_vector = [
                    metrics['avg_score'],
                    metrics['completion_rate'],
                    metrics['consistency_score'],
                    metrics['total_time_spent'],
                    metrics['topics_attempted'],
                    {'low': 1, 'medium': 2, 'high': 3}[metrics['engagement_level']],
                    metrics['improvement_trend']
                ]
                features.append(feature_vector)
                student_ids.append(student_id)
            
            if len(features) < 2:
                return {'clusters': {}, 'cluster_info': {}}
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Determine optimal number of clusters
            n_clusters = min(4, len(features))  # Max 4 clusters
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Organize results
            clusters = {}
            for i, student_id in enumerate(student_ids):
                cluster_id = int(cluster_labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(student_id)
            
            # Analyze cluster characteristics
            cluster_info = {}
            for cluster_id, student_list in clusters.items():
                cluster_metrics = [self.student_metrics[sid] for sid in student_list]
                
                cluster_info[cluster_id] = {
                    'size': len(student_list),
                    'avg_score': np.mean([m['avg_score'] for m in cluster_metrics]),
                    'avg_completion_rate': np.mean([m['completion_rate'] for m in cluster_metrics]),
                    'avg_engagement': np.mean([{'low': 1, 'medium': 2, 'high': 3}[m['engagement_level']] for m in cluster_metrics]),
                    'dominant_risk_level': max(set([m['risk_level'] for m in cluster_metrics]), key=[m['risk_level'] for m in cluster_metrics].count),
                    'characteristics': self._describe_cluster(cluster_metrics)
                }
            
            self.learning_clusters = {'clusters': clusters, 'cluster_info': cluster_info}
            return self.learning_clusters
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            return {'clusters': {}, 'cluster_info': {}}
    
    def _describe_cluster(self, cluster_metrics: List[Dict]) -> str:
        """Generate description for a cluster"""
        avg_score = np.mean([m['avg_score'] for m in cluster_metrics])
        avg_completion = np.mean([m['completion_rate'] for m in cluster_metrics])
        dominant_engagement = max(set([m['engagement_level'] for m in cluster_metrics]), 
                                key=[m['engagement_level'] for m in cluster_metrics].count)
        
        if avg_score >= 15 and avg_completion >= 0.7:
            return f"High performers with {dominant_engagement} engagement"
        elif avg_score >= 10 and avg_completion >= 0.5:
            return f"Average performers with {dominant_engagement} engagement"
        elif avg_completion < 0.3:
            return f"Low completion rate group with {dominant_engagement} engagement"
        else:
            return f"Struggling students with {dominant_engagement} engagement"
    
    def generate_performance_insights(self) -> Dict:
        """Generate overall performance insights"""
        insights = {
            'total_students': len(self.student_metrics),
            'performance_distribution': {
                'excellent': 0,
                'good': 0,
                'average': 0,
                'struggling': 0
            },
            'risk_distribution': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'engagement_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'overall_metrics': {
                'avg_score': 0,
                'avg_completion_rate': 0,
                'avg_time_spent': 0
            },
            'trends': {
                'improving_students': 0,
                'declining_students': 0,
                'stable_students': 0
            }
        }
        
        if not self.student_metrics:
            return insights
        
        # Calculate distributions
        for metrics in self.student_metrics.values():
            insights['performance_distribution'][metrics['performance_category']] += 1
            insights['risk_distribution'][metrics['risk_level']] += 1
            insights['engagement_distribution'][metrics['engagement_level']] += 1
            
            # Trends
            if metrics['improvement_trend'] > 0.2:
                insights['trends']['improving_students'] += 1
            elif metrics['improvement_trend'] < -0.2:
                insights['trends']['declining_students'] += 1
            else:
                insights['trends']['stable_students'] += 1
        
        # Overall metrics
        all_metrics = list(self.student_metrics.values())
        insights['overall_metrics'] = {
            'avg_score': np.mean([m['avg_score'] for m in all_metrics]),
            'avg_completion_rate': np.mean([m['completion_rate'] for m in all_metrics]),
            'avg_time_spent': np.mean([m['total_time_spent'] for m in all_metrics])
        }
        
        self.performance_insights = insights
        return insights
    
    def create_improvement_recommendations(self) -> Dict:
        """Create personalized improvement recommendations for each student"""
        recommendations = {}
        
        for student_id, metrics in self.student_metrics.items():
            student_recs = self._generate_student_recommendations(student_id, metrics)
            recommendations[student_id] = student_recs
        
        self.improvement_recommendations = recommendations
        return recommendations
    
    def _generate_student_recommendations(self, student_id: str, metrics: Dict) -> Dict:
        """Generate personalized recommendations for a student"""
        recommendations = {
            'priority_actions': [],
            'study_tips': [],
            'resource_suggestions': [],
            'timeline': 'immediate',
            'focus_areas': []
        }
        
        # Priority actions based on risk level
        if metrics['risk_level'] in ['critical', 'high']:
            recommendations['priority_actions'].extend([
                'Schedule immediate intervention meeting',
                'Implement daily progress monitoring',
                'Provide additional tutoring support'
            ])
            recommendations['timeline'] = 'immediate'
        
        # Specific recommendations based on issues
        if metrics['avg_score'] < 10:
            recommendations['priority_actions'].append('Focus on fundamental concept review')
            recommendations['study_tips'].extend([
                'Break study sessions into smaller, manageable chunks',
                'Use active recall techniques instead of passive reading',
                'Practice with easier problems before attempting difficult ones'
            ])
            recommendations['focus_areas'].append('Basic concept mastery')
        
        if metrics['completion_rate'] < 0.5:
            recommendations['priority_actions'].append('Improve assignment completion habits')
            recommendations['study_tips'].extend([
                'Set specific daily study goals',
                'Use a study planner to track assignments',
                'Reward yourself for completing tasks'
            ])
            recommendations['focus_areas'].append('Task completion')
        
        if metrics['consistency_score'] < 0.3:
            recommendations['study_tips'].extend([
                'Establish a regular study routine',
                'Identify and address knowledge gaps systematically',
                'Practice similar problems repeatedly until mastery'
            ])
            recommendations['focus_areas'].append('Performance consistency')
        
        if metrics['engagement_level'] == 'low':
            recommendations['priority_actions'].append('Increase learning engagement')
            recommendations['study_tips'].extend([
                'Find connections between topics and real-world applications',
                'Join study groups or find a study partner',
                'Use multimedia resources (videos, interactive tools)'
            ])
            recommendations['resource_suggestions'].extend([
                'Educational videos and tutorials',
                'Interactive learning platforms',
                'Peer study groups'
            ])
            recommendations['focus_areas'].append('Learning engagement')
        
        if metrics['improvement_trend'] < -0.2:
            recommendations['priority_actions'].append('Address declining performance trend')
            recommendations['study_tips'].extend([
                'Review recent mistakes and learn from them',
                'Identify what changed in your study approach',
                'Seek help from instructors or tutors'
            ])
            recommendations['focus_areas'].append('Performance recovery')
        
        # Topic-specific recommendations
        student_scores = self.scores_df[self.scores_df['student_id'] == student_id] if not self.scores_df.empty else pd.DataFrame()
        if not student_scores.empty:
            weak_topics = student_scores[student_scores['score'] < student_scores['score'].mean()]['topic'].unique()
            for topic in weak_topics:
                recommendations['resource_suggestions'].append(f'Additional practice materials for {topic.title()}')
                recommendations['focus_areas'].append(f'{topic.title()} improvement')
        
        # Timeline adjustment
        if metrics['risk_level'] == 'critical':
            recommendations['timeline'] = 'immediate (within 1 week)'
        elif metrics['risk_level'] == 'high':
            recommendations['timeline'] = 'urgent (within 2 weeks)'
        elif metrics['risk_level'] == 'medium':
            recommendations['timeline'] = 'short-term (within 1 month)'
        else:
            recommendations['timeline'] = 'ongoing monitoring'
        
        return recommendations
    
    def get_student_dashboard_data(self, student_id: str = None) -> Dict:
        """Get comprehensive dashboard data for all students or specific student"""
        if student_id:
            student_id = student_id.lower()
            if student_id not in self.student_metrics:
                return {'error': f'Student {student_id} not found'}
            
            return {
                'student_metrics': self.student_metrics[student_id],
                'recommendations': self.improvement_recommendations.get(student_id, {}),
                'cluster_info': self._get_student_cluster_info(student_id)
            }
        
        # Return overview for all students
        return {
            'overview': self.performance_insights,
            'at_risk_students': self.at_risk_students,
            'learning_clusters': self.learning_clusters,
            'all_students': self.student_metrics,
            'recommendations': self.improvement_recommendations
        }
    
    def _get_student_cluster_info(self, student_id: str) -> Dict:
        """Get cluster information for a specific student"""
        for cluster_id, students in self.learning_clusters['clusters'].items():
            if student_id in students:
                return {
                    'cluster_id': cluster_id,
                    'cluster_size': len(students),
                    'cluster_characteristics': self.learning_clusters['cluster_info'][cluster_id]['characteristics'],
                    'peers': [s for s in students if s != student_id]
                }
        return {}
    
    def export_analytics_report(self, output_path: str = None) -> str:
        """Export comprehensive analytics report"""
        try:
            if output_path is None:
                output_path = f"student_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'overview': self.performance_insights,
                'student_metrics': self.student_metrics,
                'at_risk_analysis': self.at_risk_students,
                'learning_clusters': self.learning_clusters,
                'recommendations': self.improvement_recommendations
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"Analytics report exported to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error exporting analytics report: {e}")
            return ""

# Example usage and testing
if __name__ == "__main__":
    # Initialize analytics system
    analytics = StudentPerformanceAnalytics()
    
    print("=== Student Performance Analytics ===")
    
    # Get overview
    overview = analytics.get_student_dashboard_data()
    print(f"Total students analyzed: {overview['overview']['total_students']}")
    print(f"At-risk students: {len(overview['at_risk_students']['critical']) + len(overview['at_risk_students']['high'])}")
    
    # Export report
    analytics.export_analytics_report()
