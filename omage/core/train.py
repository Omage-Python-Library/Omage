"""
Omage Train — Training utilities
"""


def train(model, data, epochs=10, batch_size=32, verbose=True):
    """
    Simple training function
    
    Example:
        import omage as og
        model = og.model(type="classifier", layers=[128, 10])
        og.train(model, data, epochs=10)
    """
    return model.train(data, epochs=epochs, batch_size=batch_size, verbose=verbose)