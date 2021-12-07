from jax import value_and_grad, vmap, numpy as jnp


def two_dimensional_multicubic(f, N_res):
    """
    Integrations a function over unit square with `N_res` equal partitions.

    Applies a multi-cubic appoximation, f_approx(x) = vec(poly(x,3)^T @ poly(y,3)) @ beta

    where poly(x,3) are third order polynomials in x.

    Args:
        f: callable(x,y)
        N_res: number of equal partitions

    Returns:
        int_0^1 int_0^1 f(x,y) dx dy
    """
    dx = dy = 1./(N_res+1.)

    x = y = jnp.linspace(0., 1., N_res+1)

    X, Y = jnp.meshgrid(x,y, indexing='ij')
    
    X_f_agg = []
    Y_f_agg = []
    f_coefficient = []
    def add_f_coord(X,Y,coefficient):
        X_f_agg.append(jnp.ravel(X))
        Y_f_agg.append(jnp.ravel(Y))
        f_coefficient.append(coefficient * jnp.ones_like(Y_f_agg[-1]))

    X_fx_agg = []
    Y_fx_agg = []
    fx_coefficient = []

    def add_fx_coord(X, Y, coefficient):
        X_fx_agg.append(jnp.ravel(X))
        Y_fx_agg.append(jnp.ravel(Y))
        fx_coefficient.append(coefficient * jnp.ones_like(Y_fx_agg[-1]))

    X_fy_agg = []
    Y_fy_agg = []
    fy_coefficient = []

    def add_fy_coord(X, Y, coefficient):
        X_fy_agg.append(jnp.ravel(X))
        Y_fy_agg.append(jnp.ravel(Y))
        fy_coefficient.append(coefficient * jnp.ones_like(Y_fy_agg[-1]))

    X_fxy_agg = []
    Y_fxy_agg = []
    fxy_coefficient = []

    def add_fxy_coord(X, Y, coefficient):
        X_fxy_agg.append(jnp.ravel(X))
        Y_fxy_agg.append(jnp.ravel(Y))
        fxy_coefficient.append(coefficient * jnp.ones_like(Y_fxy_agg[-1]))
        
    # interior sum
    X_interior, Y_interior = X[1:-1,1:-1], Y[1:-1,1:-1]
    add_f_coord(X_interior,Y_interior,1)
    
    X_left, Y_left = X[0, 1:-1], Y[0, 1:-1]
    add_f_coord(X_left, Y_left, 0.5)
    add_fx_coord(X_left, Y_left, (1./12.) * dx)

    X_right, Y_right = X[-1, 1:-1], Y[-1, 1:-1]
    add_f_coord(X_right, Y_right, 0.5)
    add_fx_coord(X_right, Y_right, (-1. / 12.) * dx)

    X_bottom, Y_bottom = X[1:-1, 0], Y[1:-1, 0]
    add_f_coord(X_bottom, Y_bottom, 0.5)
    add_fy_coord(X_bottom, Y_bottom, (1./12.) * dx)

    X_top, Y_top = X[1:-1, -1], Y[1:-1, -1]
    add_f_coord(X_top, Y_top, 0.5 )
    add_fy_coord(X_top, Y_top, (-1. / 12.) * dx)

    X_topleft, Y_topleft = X[0,-1], Y[0,-1]
    add_f_coord(X_topleft, Y_topleft, 0.25)
    add_fx_coord(X_topleft, Y_topleft, (1./24.) * dx)
    add_fy_coord(X_topleft, Y_topleft, (-1./24.) * dx)
    add_fxy_coord(X_topleft, Y_topleft, (-1./144.) * dx**2)

    X_topright, Y_topright = X[-1, -1], Y[-1, -1]
    add_f_coord(X_topright, Y_topright, 0.25)
    add_fx_coord(X_topright, Y_topright, (-1. / 24.) * dx)
    add_fy_coord(X_topright, Y_topright, (-1. / 24.) * dx)
    add_fxy_coord(X_topright, Y_topright, (1. / 144.) * dx ** 2)

    X_bottomleft, Y_bottomleft = X[0, 0], Y[0, 0]
    add_f_coord(X_bottomleft, Y_bottomleft, 0.25 )
    add_fx_coord(X_bottomleft, Y_bottomleft, (1. / 24.) * dx)
    add_fy_coord(X_bottomleft, Y_bottomleft, (1. / 24.) * dx)
    add_fxy_coord(X_bottomleft, Y_bottomleft, (1. / 144.) * dx ** 2)

    X_bottomright, Y_bottomright = X[-1, 0], Y[-1, 0]
    add_f_coord(X_bottomright, Y_bottomright, 0.25)
    add_fx_coord(X_bottomright, Y_bottomright, (-1. / 24.) * dx)
    add_fy_coord(X_bottomright, Y_bottomright, (1. / 24.) * dx)
    add_fxy_coord(X_bottomright, Y_bottomright, (-1. / 144.) * dx ** 2)

    X_f_agg = jnp.concatenate(X_f_agg)
    Y_f_agg = jnp.concatenate(Y_f_agg)
    f_coefficient = jnp.concatenate(f_coefficient)

    X_fx_agg = jnp.concatenate(X_fx_agg)
    Y_fx_agg = jnp.concatenate(Y_fx_agg)
    fx_coefficient = jnp.concatenate(fx_coefficient)

    X_fy_agg = jnp.concatenate(X_fy_agg)
    Y_fy_agg = jnp.concatenate(Y_fy_agg)
    fy_coefficient = jnp.concatenate(fy_coefficient)

    X_fxy_agg = jnp.concatenate(X_fxy_agg)
    Y_fxy_agg = jnp.concatenate(Y_fxy_agg)
    fxy_coefficient = jnp.concatenate(fxy_coefficient)

    fx = lambda x,y: value_and_grad(f, argnums=0)(x,y)[1]
    fy = lambda x,y: value_and_grad(f, argnums=1)(x,y)[1]
    fxy = lambda x, y: value_and_grad(lambda x, y: fx(x, y), argnums=1)(x,y)[1]

    f_sum = jnp.sum(vmap(f)(X_f_agg, Y_f_agg) * f_coefficient)
    fx_sum = jnp.sum(vmap(fx)(X_fx_agg, Y_fx_agg) * fx_coefficient)
    fy_sum = jnp.sum(vmap(fy)(X_fy_agg, Y_fy_agg) * fy_coefficient)
    fxy_sum = jnp.sum(vmap(fxy)(X_fxy_agg, Y_fxy_agg) * fxy_coefficient)

    return (f_sum + fx_sum + fy_sum + fxy_sum)*dx**2


def test_two_dimensional_multicubic():
    def f(x,y):
        return jnp.exp((-x**2 + y**2))


    x = y = jnp.linspace(0., 1., 100 + 1)
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    f_val = vmap(f)(X.flatten(), Y.flatten()).reshape(X.shape)
    import pylab as plt
    plt.imshow(f_val)
    plt.colorbar()
    plt.show()

    def f_int(N_res):
        dx = dy = 1. / (N_res + 1.)

        x = y = jnp.linspace(0., 1., N_res + 1)

        X, Y = jnp.meshgrid(x, y, indexing='ij')

        f_val = vmap(f)(X.flatten(), Y.flatten()).reshape(X.shape)

        return f_val.sum() * dx ** 2


    print(f"f integral high-res.: {f_int(10)}")

    print(two_dimensional_multicubic(f, 10))

    for n in range(10,201,20):
        plt.scatter(n,two_dimensional_multicubic(f, n))
        plt.scatter(n,f_int(n))
    plt.axhline(1.092)
    plt.yscale('log')
    plt.show()

    