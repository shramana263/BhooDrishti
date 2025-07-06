import ee
import time
from datetime import datetime

def monitor_export_tasks(task_ids, check_interval=30):
    """Monitor the progress of export tasks"""
    print("📋 Monitoring export tasks...")
    
    while task_ids:
        completed_tasks = []
        
        for task_id in task_ids:
            try:
                task = ee.batch.Task(task_id)
                status = task.status()
                
                print(f"Task {task_id}: {status['state']}")
                
                if status['state'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    completed_tasks.append(task_id)
                    
                    if status['state'] == 'COMPLETED':
                        print(f"✅ Task {task_id} completed successfully")
                    else:
                        print(f"❌ Task {task_id} failed: {status.get('error_message', 'Unknown error')}")
                        
            except Exception as e:
                print(f"Error checking task {task_id}: {e}")
                completed_tasks.append(task_id)
        
        # Remove completed tasks
        for task_id in completed_tasks:
            task_ids.remove(task_id)
        
        if task_ids:
            print(f"⏳ {len(task_ids)} tasks still running. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)
        else:
            print("🎉 All export tasks completed!")

def get_task_list():
    """Get list of all tasks for the authenticated user"""
    tasks = ee.batch.Task.list()
    
    print("📋 Recent Export Tasks:")
    for i, task in enumerate(tasks[:10]):  # Show last 10 tasks
        status = task.status()
        print(f"{i+1}. {status['description']}: {status['state']}")
        if 'start_timestamp_ms' in status:
            start_time = datetime.fromtimestamp(status['start_timestamp_ms'] / 1000)
            print(f"   Started: {start_time}")