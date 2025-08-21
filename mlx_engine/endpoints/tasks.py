"""Task management endpoint module."""

from flask import jsonify
from .base import EndpointBase

class TaskEndpoint(EndpointBase):
    """Task management endpoint implementation."""
    
    def get_task_status(self, task_id):
        """Get the status of a background task."""
        with self.api.task_lock:
            if task_id in self.api.running_tasks:
                return jsonify({
                    'task_id': task_id,
                    'status': 'running',
                    'started_at': self.api.running_tasks[task_id].get('started_at'),
                    'description': self.api.running_tasks[task_id].get('description')
                })
            elif task_id in self.api.task_results:
                result = self.api.task_results[task_id]
                return jsonify({
                    'task_id': task_id,
                    'status': result['status'],
                    'result': result['result'] if result['status'] == 'completed' else None,
                    'error': result['error'] if result['status'] == 'failed' else None
                })
            else:
                return jsonify({'error': 'Task not found'}), 404
    
    def list_tasks(self):
        """List all tasks (running and completed)."""
        with self.api.task_lock:
            tasks = []
            for task_id, task_info in self.api.running_tasks.items():
                tasks.append({
                    'task_id': task_id,
                    'status': 'running',
                    'started_at': task_info.get('started_at'),
                    'description': task_info.get('description')
                })
            for task_id, result in self.api.task_results.items():
                tasks.append({
                    'task_id': task_id,
                    'status': result['status'],
                    'completed_at': result.get('completed_at')
                })
            return jsonify({'tasks': tasks})
    
    def debug_routes(self):
        """Debug endpoint to list all registered routes."""
        routes = []
        for rule in self.app.url_map.iter_rules():
            methods = ','.join(rule.methods)
            routes.append({
                'endpoint': rule.endpoint,
                'methods': methods,
                'rule': str(rule)
            })
        return jsonify({'routes': routes})

def register_task_routes(api_instance):
    """Register task routes with the Flask app."""
    task_endpoint = TaskEndpoint(api_instance)
    
    api_instance.app.route('/v1/tasks/status/<task_id>', methods=['GET'])(task_endpoint.get_task_status)
    api_instance.app.route('/v1/tasks', methods=['GET'])(task_endpoint.list_tasks)
    api_instance.app.route('/debug/routes', methods=['GET'])(task_endpoint.debug_routes)