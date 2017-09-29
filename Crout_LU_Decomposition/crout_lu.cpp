#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <stdexcept>

using std::swap;

const double eps = 0.00000001;

// matrix class
class matrix {
    int r; // number of rows
    int c; // number of columns
    double** elems = NULL; // container for the matrix elements

public:
    matrix() {
    }

    // construct (n, m) sized matrix filled with zeros
    matrix(int n, int m) {
        r = n;
        c = m;

        elems = new double*[r];

        for (int i = 0; i < r; ++i) {
            elems[i] = new double[c];
        }

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                elems[i][j] = 0.0;
            }
        }
    }

    matrix(const matrix &m) {
        r = m.r;
        c = m.c;

        elems = new double*[r];

        for (int i = 0; i < r; ++i) {
            elems[i] = new double[c];
        }

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                elems[i][j] = m.elems[i][j];
            }
        }
    }
    
    void print() const {
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                printf("%.5f ", elems[i][j]);
            }
            printf("\n");
        }
    }

    ~matrix() {
        if (elems != NULL) {
            for (int i = 0; i < r; ++i) {
                if (elems[i] != NULL) {
                    delete elems[i];
                }
            }
            delete elems;
        }
    }

    // operator for accessing the matrix
    double* operator [](int i) const {
        return elems[i];
    }

    double*& operator [](int i) {
        return elems[i];
    }

    matrix& operator = (const matrix &other) {
        if (this != &other) {
            matrix cp(other);
            swap(r, cp.r);
            swap(c, cp.c);
            swap(elems, cp.elems);
        }
        return *this;
    }

    // matrix multiplication operator
    matrix operator * (const matrix &other) {
        int n = this->num_rows(), m = other.num_cols();

        if (this->num_cols() != other.num_rows()) {
            throw std::invalid_argument("incompatible matrix dimensions");
        }

        matrix res(n, m);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                res[i][j] = 0.0;
                for (int k = 0; k < this->num_cols(); ++k) {
                    res[i][j] += (*this)[i][k] * other[k][j];
                }
            }
        }

        return res;
    }

    unsigned int num_rows() const {
        return r;
    }

    unsigned int num_cols() const {
        return c;
    }
};

class LU_Container {
    matrix L; // the L matrix
    matrix U; // the U matrix
    matrix A; // the A matrix
    int* P; // keeps track of row interchanges
    int n; // number of rows and columns of A


public:
    LU_Container(const matrix &a) {
        if (a.num_rows() != a.num_cols()) {
            throw std::invalid_argument("incompatible matrix dimensions");
        }
        A = a;
        n = A.num_rows();
        L = matrix(n, n);
        U = matrix(n, n);

        P = new int[n];
        for (int i = 0; i < n; ++i) {
            P[i] = i;
        }

        decompose();
    }

    matrix getL() {
        return L;
    }

    matrix getU() {
        return U;
    }

    matrix solve(const matrix &b) {
        // permute b
        matrix B(n, 1);
        for (int i = 0; i < n; ++i) {
            B[i][0] = b[P[i]][0];
        }

        // solve Ly = B
        matrix y(n, 1);
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int j = 0; j < i; ++j) {
                s += L[i][j] * y[j][0];
            }
            y[i][0] = (B[i][0] - s) / L[i][i];
        }

        // solve Ux = y
        matrix x(n, 1);
        for (int i = n - 1; i >= 0; --i) {
            double s = 0;
            for (int j = i + 1; j < n; ++j) {
                s += U[i][j] * x[j][0];
            }
            x[i][0] = y[i][0] - s;
        }

        return x;
    }

    double cond_number() {
        A = L * U;
        double l2_norm = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                l2_norm += A[i][j] * A[i][j];
            }
        }
        l2_norm = sqrt(l2_norm);

        double abs_det = 1;

        for (int i = 0; i < n; ++i) {
            abs_det *= L[i][i];
        }

        if (abs_det < 0) {
            abs_det = -abs_det;
        }

        return l2_norm / abs_det;
    }

    ~LU_Container() {
        delete P;
    }

private:
    void decompose() {
        for (int i = 0; i < n; ++i) {
            U[i][i] = 1.0;
        }

        double* S = new double[n]; // max abs value for each row
        for (int i = 0; i < n; ++i) {
            S[i] = -1;
            for (int j = 0; j < n; ++j) {
                if (A[i][j] > S[i]) {
                    S[i] = A[i][j];
                }
                if (-A[i][j] > S[i]) {
                    S[i] = -A[i][j];
                }
            }
        }

        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                double s = 0;
                for (int p = 0; p < j; ++p) {
                    s += L[i][p] * U[p][j];
                }
                L[i][j] = A[i][j] - s;
            }

            // do pivoting if needed
            int maxind = -1;
            double maxval = -1;

            for (int i = j; i < n; ++i) {
                if (L[i][j] / S[i] > maxval) {
                    maxval = L[i][j] / S[i];
                    maxind = i;
                }
                if (-L[i][j] / S[i] > maxval) {
                    maxval = -L[i][j] / S[i];
                    maxind = i;
                }
            }

            if (maxind != j) {
                for (int i = j + 1; i < n; ++i) {
                    swap(A[j][i], A[maxind][i]);
                }

                for (int i = 0; i <= j; ++i) {
                    swap(L[j][i], L[maxind][i]);
                }

                swap(S[j], S[maxind]);
                swap(P[j], P[maxind]);
            }

            for (int i = j + 1; i < n; ++i) {
                double s = 0;
                for (int p = 0; p < j; ++p) {
                    s += L[j][p] * U[p][i];
                }
                U[j][i] = (A[j][i] - s) / L[j][j];
            }
        }

        delete S;
    }
};

int main() {

    int n = 8;
    matrix A(n, n);

    double tmp[n][n] = {
        {17, 1, -1, 2, -2, 3, -3, 4},
        {2, -16, -1, 3, -2, 1, 1, -4},
        {-1, 1, 15, 2, -1, 2, -1, 1},
        {2, 4, 1, -14, 1, 3, 4, -1},
        {1, 3, 1, -1, 13, 1, -2, 3},
        {-2, 1, 2, -1, 2, -12, -1, 1},
        {3, 4, -1, 1, 2, -2, 11, -3},
        {2, 1, 1, 1, -1, 1, -2, -10}
    };

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = tmp[i][j];
        }
    }

    LU_Container lu = LU_Container(A);

    lu.getL().print();
    printf("\n");
    lu.getU().print();

    /*
    17.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 
    2.00000 -16.11765 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 
    -1.00000 1.05882 14.88321 0.00000 0.00000 0.00000 0.00000 0.00000 
    2.00000 3.88235 0.90511 -13.70917 0.00000 0.00000 0.00000 0.00000 
    1.00000 2.94118 0.89781 -0.75184 12.82149 0.00000 0.00000 0.00000 
    -2.00000 1.11765 1.82117 -0.85434 1.73811 -12.12964 0.00000 0.00000 
    3.00000 3.82353 -1.03285 1.46248 1.94314 -2.03755 12.74801 0.00000 
    2.00000 0.88235 1.06934 0.75086 -0.72420 0.70654 -1.40384 -11.35592

    1.00000 0.05882 -0.05882 0.11765 -0.11765 0.17647 -0.17647 0.23529 
    0.00000 1.00000 0.05474 -0.17153 0.10949 -0.04015 -0.08394 0.27737 
    0.00000 0.00000 1.00000 0.15449 -0.08288 0.14909 -0.07308 0.06327 
    0.00000 0.00000 0.00000 1.00000 -0.06457 -0.19461 -0.34612 0.19000 
    0.00000 0.00000 0.00000 0.00000 1.00000 0.05159 -0.13815 0.15871 
    0.00000 0.00000 0.00000 0.00000 0.00000 1.00000 0.09742 -0.07682 
    0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 1.00000 -0.42704 
    0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 1.00000
    */

    matrix b(n, 1);

    double tmpb[n] = {33, 30, -24, -30, 25, 22, -27, 18};
    for (int i = 0; i < n; ++i) {
        b[i][0] = tmpb[i];
    }

    matrix solution = lu.solve(b);

    printf("solution:\n");
    solution.print();

    /*
    2.16012 
    -1.72302 
    -1.17225 
    0.63389 
    2.17523 
    -2.02120 
    -3.69212 
    -1.27533
    */

    double cond_number = lu.cond_number();

    printf("\ncondition number = %.9f\n", cond_number);
    // 0.000000033

    return 0;
}