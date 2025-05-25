class minimizationAlgorithms:
    def __init__(self, algorithm):
        self.algorithm = algorithm # either gradient descent OR newton search directions
        self.history = []
        
    def minimize(self, f, x0, obj_tol, param_tol, max_iter):
        if self.algorithm == 'Gradient Descent':
            self.gradient_descent(f, x0, obj_tol, param_tol, max_iter)
        elif self.algorithm == 'Newton Search Directions':
            self.newton_search_directions(f, x0, obj_tol, param_tol, max_iter)
        else:
            print("Invalid Selection")

    
    def gradient_descent(self, f, x0, obj_tol, param_tol, max_iter):
        pass

    def newton_search_directions(self, f, x0, obj_tol, param_tol, max_iter):
        pass