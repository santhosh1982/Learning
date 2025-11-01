# Comprehensive Linear Algebra: Theory and Practice

## Table of Contents

1. [Introduction](#introduction)
2. [Chapter 1: Foundations and Basic Concepts](#chapter-1)
3. [Chapter 2: Systems of Linear Equations](#chapter-2)
4. [Chapter 3: Vectors and Vector Spaces](#chapter-3)
5. [Chapter 4: Matrices and Matrix Operations](#chapter-4)
6. [Chapter 5: Determinants](#chapter-5)
7. [Chapter 6: Vector Spaces and Subspaces](#chapter-6)
8. [Chapter 7: Linear Independence, Basis, and Dimension](#chapter-7)
9. [Chapter 8: Linear Transformations](#chapter-8)
10. [Chapter 9: Eigenvalues and Eigenvectors](#chapter-9)
11. [Chapter 10: Diagonalization](#chapter-10)
12. [Chapter 11: Inner Product Spaces](#chapter-11)
13. [Chapter 12: Applications and Advanced Topics](#chapter-12)

---

## Introduction

Linear algebra is a branch of mathematics that studies vectors, matrices, linear functions, and vector spaces. It forms the mathematical foundation for numerous applications in science, engineering, computer science, and economics. The power of linear algebra lies in its ability to organize and simplify complex multidimensional problems, allowing us to solve systems involving thousands or even millions of variables.

### Why Linear Algebra Matters

**Historical Context**: Linear algebra emerged from the study of systems of linear equations. Over centuries, mathematicians developed increasingly sophisticated techniques to solve these systems, culminating in the abstract framework we use today.

**Practical Applications**: From computer graphics and machine learning to quantum mechanics and structural engineering, linear algebra provides essential tools. Modern applications include:

- Machine Learning: Neural networks and deep learning rely on matrix operations
- Computer Graphics: Transformations like rotation, scaling, and projection use matrices
- Data Science: Dimensionality reduction and principal component analysis employ eigenvectors
- Physics: Quantum mechanics is fundamentally linear algebraic
- Economics: Input-output models and optimization use linear systems
- Engineering: Finite element analysis and structural mechanics depend on linear algebra

### Core Philosophy

Linear algebra is the study of vectors and linear functions. A vector is an object you can add and scalar multiply. A linear function respects vector addition and scalar multiplication. This simple definition enables powerful computational and theoretical frameworks.

---

## Chapter 1: Foundations and Basic Concepts

### 1.1 What is Linear Algebra?

Linear algebra organizes mathematical information in structured ways. Consider a portfolio containing stocks from multiple companies. The value depends on:
- Number of shares in each company
- Current price of each stock

We need to organize this information so we can efficiently compute total portfolio value or find what holdings produce a target value.

This organizational principle extends far beyond finance. Whenever we have linear relationships between quantities, linear algebra provides systematic methods for analysis.

### 1.2 Scalars

**Definition**: A scalar is a single real number that has magnitude but no direction.

Examples of scalars include:
- Prices: $15.50 per share
- Temperatures: 25°C
- Weights: 73.5 kg

Scalars can be added, subtracted, multiplied, and divided (except by zero) following familiar rules of arithmetic.

### 1.3 Vectors

**Definition**: A vector is an ordered list of numbers (scalars) that can be added and scaled.

#### Vector Notation

Vectors are typically written in one of two forms:

**Column Vector**: \(\begin{pmatrix} x \\ y \\ z \end{pmatrix}\)

**Row Vector**: \((x, y, z)\) or \([x, y, z]\)

We denote vectors with bold lowercase letters (\(\mathbf{v}\)) or with arrows (\(\vec{v}\)).

#### Examples of Vectors

1. **2-vector** (point in a plane): \(\begin{pmatrix} 3 \\ 4 \end{pmatrix}\) represents the point 3 units right and 4 units up

2. **3-vector** (point in space): \(\begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}\) represents a point in 3D space

3. **n-vector**: \(\begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}\) represents a point in n-dimensional space

4. **Time Series**: \(\begin{pmatrix} 100 \\ 105 \\ 108 \\ 110 \end{pmatrix}\) represents stock prices over four days

#### The Zero Vector

The zero vector **0** has all components equal to zero: \(\begin{pmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{pmatrix}\)

This acts as the identity element for vector addition.

### 1.4 Vector Operations

#### Vector Addition

When vectors have the same dimension, they can be added component-wise.

**Definition**: \(\begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix} + \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix} = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{pmatrix}\)

**Geometric Interpretation**: Vector addition can be visualized as placing vectors head-to-tail.

**Properties of Vector Addition**:
- Commutativity: **u** + **v** = **v** + **u**
- Associativity: (**u** + **v**) + **w** = **u** + (**v** + **w**)
- Identity: **v** + **0** = **v**
- Inverse: **v** + (−**v**) = **0**

#### Scalar Multiplication

Multiplying a vector by a scalar scales each component.

**Definition**: \(c\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} cx \\ cy \\ cz \end{pmatrix}\)

**Geometric Interpretation**: Scalar multiplication scales the vector's length. If c > 1, the vector stretches; if 0 < c < 1, it shrinks; if c < 0, it reverses direction.

**Properties of Scalar Multiplication**:
- Distributivity: \(c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}\)
- Associativity: \((cd)\mathbf{v} = c(d\mathbf{v})\)
- Identity: \(1 \cdot \mathbf{v} = \mathbf{v}\)

#### Linear Combinations

A linear combination of vectors is a sum of scalar multiples of those vectors.

**Definition**: If **v₁**, **v₂**, ..., **vₖ** are vectors and c₁, c₂, ..., cₖ are scalars, then

\[c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k\]

is a linear combination of these vectors.

**Example**: 2**u** + 3**v** − **w** is a linear combination of vectors **u**, **v**, and **w**.

### 1.5 Different Types of Vectors

Linear algebra unifies different mathematical objects that behave like vectors:

1. **n-tuples of real numbers**: \(\begin{pmatrix} 2.5 \\ -1.3 \\ 7.8 \end{pmatrix}\)

2. **Polynomials**: \(p(x) = 3 + 2x + x^2\) can be represented as \(\begin{pmatrix} 3 \\ 2 \\ 1 \end{pmatrix}\)

3. **Functions**: The set of all continuous functions on [0,1] forms a vector space

4. **Matrices**: Can be viewed as vectors when appropriately stacked

5. **Complex numbers**: The set of all complex numbers forms a vector space over the reals

The key insight is that the same abstract theory applies to all these different types of vectors because they all satisfy the same fundamental properties of vector addition and scalar multiplication.

---

## Chapter 2: Systems of Linear Equations

### 2.1 Introduction to Linear Systems

Many practical problems reduce to solving systems of linear equations. The goal is to find values of variables that simultaneously satisfy multiple linear constraints.

**Standard Form**: A system of m linear equations in n variables has the form:

\[\begin{align}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{align}\]

where:
- The \(a_{ij}\) are coefficients
- The \(b_i\) are constant terms
- The \(x_i\) are variables we want to solve for

### 2.2 Geometric Interpretation

#### Two Variables (2D)

Each linear equation in two variables represents a line in the plane.

**Possible Outcomes**:
1. **Unique Solution**: Lines intersect at one point
2. **No Solution**: Lines are parallel (inconsistent system)
3. **Infinitely Many Solutions**: Lines coincide (dependent system)

#### Three Variables (3D)

Each linear equation in three variables represents a plane in 3D space.

**Possible Outcomes**:
1. **Unique Solution**: Planes intersect at a single point
2. **No Solution**: Planes have no common intersection
3. **Infinitely Many Solutions**: Planes intersect in a line or are identical

#### General Case

For n variables in m equations:
- If \(m < n\): Usually infinitely many solutions or no solution (rarely unique)
- If \(m = n\): Often a unique solution
- If \(m > n\): Often no solution (overdetermined system)

### 2.3 Elementary Operations and Equivalence

**Definition**: Two systems are equivalent if they have exactly the same solution set.

**Elementary Operations** (preserve equivalence):
1. **Swap equations**: Reorder the equations
2. **Scale an equation**: Multiply an equation by a nonzero scalar
3. **Add a multiple**: Add a scalar multiple of one equation to another

These operations never change the solution set because they correspond to reversible algebraic manipulations.

### 2.4 Gaussian Elimination

**Gaussian Elimination** systematically uses elementary operations to transform a system into a simpler form from which solutions are easy to find.

#### Procedure

1. **Forward Elimination**: Use row operations to create zeros below the diagonal, obtaining an upper triangular form

2. **Back Substitution**: Starting from the last equation, solve for each variable in terms of the others

#### Example

Given system:
\[\begin{align}
x + 3y + 6z &= 25 \\
2x + 7y + 14z &= 58 \\
2y + 5z &= 19
\end{align}\]

**Step 1**: Eliminate x from equations 2 and 3
- Equation 2 − 2×Equation 1: \(y + 2z = 8\)

**Step 2**: Eliminate y from equation 3
- Equation 3 − 2×(new Equation 2): \(z = 3\)

**Back Substitution**:
- From Eq. 3: \(z = 3\)
- From Eq. 2: \(y + 2(3) = 8 \Rightarrow y = 2\)
- From Eq. 1: \(x + 3(2) + 6(3) = 25 \Rightarrow x = 1\)

**Solution**: \((x, y, z) = (1, 2, 3)\)

### 2.5 Special Cases

#### Consistent vs. Inconsistent

- **Consistent**: System has at least one solution
- **Inconsistent**: System has no solution

An inconsistent system appears as a row of zeros equal to a nonzero constant: \(0 = 5\), which is impossible.

#### Dependent Systems

When multiple equations represent the same constraint, we have fewer independent equations than variables. This leads to infinitely many solutions (if consistent).

#### Rank and Solvability

The **rank** of the system is the number of independent equations. For a system to be:
- **Consistent with unique solution**: rank = n (number of variables)
- **Consistent with infinitely many solutions**: rank < n
- **Inconsistent**: The augmented matrix has rank greater than the coefficient matrix rank

---

## Chapter 3: Vectors and Vector Spaces

### 3.1 Geometric Vectors in \(\mathbb{R}^2\) and \(\mathbb{R}^3\)

Vectors in Euclidean space can be visualized geometrically.

#### Magnitude (Length)

The magnitude or norm of vector \(\mathbf{v} = \begin{pmatrix} x \\ y \end{pmatrix}\) is:

\[\|\mathbf{v}\| = \sqrt{x^2 + y^2}\]

In 3D: \(\|\mathbf{v}\| = \sqrt{x^2 + y^2 + z^2}\]

In n-dimensions: \(\|\mathbf{v}\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}\)

#### Direction

A unit vector has magnitude 1. The unit vector in the direction of **v** is:

\[\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}\]

### 3.2 Dot Product

**Definition**: For vectors **u** and **v** in \(\mathbb{R}^n\):

\[\mathbf{u} \cdot \mathbf{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n\]

**Properties**:
- Commutativity: **u** · **v** = **v** · **u**
- Distributivity: **u** · (**v** + **w**) = **u** · **v** + **u** · **w**
- Scalar association: \((c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v})\)

**Geometric Interpretation**:

\[\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta\]

where θ is the angle between the vectors.

**Orthogonality**: Vectors **u** and **v** are orthogonal (perpendicular) if **u** · **v** = 0.

#### Projection

The projection of **u** onto **v** is:

\[\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v}\]

### 3.3 Cross Product (in \(\mathbb{R}^3\))

The cross product of \(\mathbf{u} = \begin{pmatrix} u_1 \\ u_2 \\ u_3 \end{pmatrix}\) and \(\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ v_3 \end{pmatrix}\) is:

\[\mathbf{u} \times \mathbf{v} = \begin{pmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{pmatrix}\]

**Properties**:
- Anti-commutativity: **u** × **v** = −(**v** × **u**)
- Magnitude: \(\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\| \|\mathbf{v}\| \sin \theta\)
- The cross product is perpendicular to both **u** and **v**

**Application**: The magnitude of **u** × **v** equals the area of the parallelogram spanned by **u** and **v**.

---

## Chapter 4: Matrices and Matrix Operations

### 4.1 Introduction to Matrices

**Definition**: A matrix is a rectangular array of numbers arranged in rows and columns.

**Notation**: An m × n matrix has m rows and n columns:

\[A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}\]

**Terminology**:
- **Square matrix**: m = n
- **Row matrix**: m = 1
- **Column matrix**: n = 1
- **Diagonal matrix**: All off-diagonal entries are zero
- **Identity matrix**: Square matrix with ones on diagonal, zeros elsewhere (denoted I)
- **Zero matrix**: All entries are zero (denoted 0)

### 4.2 Matrix Operations

#### Matrix Addition and Subtraction

Matrices of the same dimensions can be added/subtracted component-wise:

\[A + B = (a_{ij} + b_{ij})\]

#### Scalar Multiplication

\[cA = (ca_{ij})\]

#### Matrix Multiplication

For matrices A (m × p) and B (p × n), the product AB is an m × n matrix where:

\[(AB)_{ij} = \sum_{k=1}^{p} a_{ik}b_{kj}\]

**Important Properties**:
- NOT commutative: AB ≠ BA in general
- Associative: (AB)C = A(BC)
- Distributive: A(B + C) = AB + AC
- Identity: AI = IA = A

**Geometric Interpretation**: Matrix multiplication represents composition of linear transformations.

#### Transpose

The transpose of A, denoted \(A^T\), is obtained by swapping rows and columns:

\[(A^T)_{ij} = a_{ji}\]

**Properties**:
- \((A^T)^T = A\)
- \((A + B)^T = A^T + B^T\)
- \((AB)^T = B^T A^T\)

### 4.3 Matrix Inverses

**Definition**: For a square matrix A, the inverse \(A^{-1}\) (if it exists) satisfies:

\[A A^{-1} = A^{-1} A = I\]

**Conditions for Invertibility**:
- Matrix must be square
- Matrix must have nonzero determinant
- Matrix rows/columns must be linearly independent

**Computing the Inverse** (for 2×2 matrices):

\[\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}\]

For larger matrices, methods include:
- Gaussian elimination with augmented identity matrix
- Adjugate matrix method
- Computational algorithms

### 4.4 Matrix Representation of Linear Systems

A system of equations:
\[\begin{align}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\
&\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m
\end{align}\]

Can be written as:

\[A\mathbf{x} = \mathbf{b}\]

where:
- A is the coefficient matrix (m × n)
- **x** is the variable vector (n × 1)
- **b** is the constant vector (m × 1)

---

## Chapter 5: Determinants

### 5.1 Definition and Computation

**Definition**: The determinant of a square matrix A, denoted det(A) or |A|, is a scalar that encodes important information about the matrix.

#### 2×2 Determinant

\[\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc\]

#### 3×3 Determinant (Rule of Sarrus or Cofactor Expansion)

\[\det \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = aei + bfg + cdh - ceg - afh - bdi\]

#### General n×n Determinant (Cofactor Expansion)

\[\det(A) = \sum_{j=1}^{n} a_{ij} C_{ij}\]

where \(C_{ij} = (-1)^{i+j} M_{ij}\) is the cofactor, and \(M_{ij}\) is the minor (determinant of the submatrix obtained by deleting row i and column j).

### 5.2 Properties of Determinants

1. **Scaling**: \(\det(cA) = c^n \det(A)\) for an n × n matrix
2. **Transpose**: \(\det(A^T) = \det(A)\)
3. **Product**: \(\det(AB) = \det(A) \det(B)\)
4. **Inverse**: \(\det(A^{-1}) = \frac{1}{\det(A)}\) if A is invertible
5. **Swap rows**: Swapping two rows changes the sign of the determinant
6. **Identical rows**: If two rows are identical, determinant is zero
7. **Row operations**: Adding a multiple of one row to another doesn't change the determinant

### 5.3 Geometric Interpretation

For a 2×2 matrix, |det(A)| is the area of the parallelogram spanned by the column vectors.

For a 3×3 matrix, |det(A)| is the volume of the parallelepiped spanned by the column vectors.

### 5.4 Applications

**Invertibility**: A square matrix A is invertible if and only if det(A) ≠ 0.

**Cramer's Rule**: For a square system Ax = b with det(A) ≠ 0:

\[x_i = \frac{\det(A_i)}{\det(A)}\]

where \(A_i\) is the matrix obtained by replacing the ith column of A with b.

---

## Chapter 6: Vector Spaces and Subspaces

### 6.1 Vector Space Axioms

**Definition**: A set V with operations of addition and scalar multiplication is a vector space over a field F if the following axioms hold for all **u**, **v**, **w** ∈ V and all c, d ∈ F:

**Closure**:
- **u** + **v** ∈ V
- c**u** ∈ V

**Addition Properties**:
1. Commutativity: **u** + **v** = **v** + **u**
2. Associativity: (**u** + **v**) + **w** = **u** + (**v** + **w**)
3. Identity element: ∃ **0** ∈ V such that **v** + **0** = **v**
4. Inverse elements: For each **v** ∈ V, ∃ −**v** ∈ V such that **v** + (−**v**) = **0**

**Scalar Multiplication Properties**:
5. Associativity: c(d**v**) = (cd)**v**
6. Identity: 1**v** = **v**
7. Distributivity (vectors): c(**u** + **v**) = c**u** + c**v**
8. Distributivity (scalars): (c + d)**v** = c**v** + d**v**

### 6.2 Examples of Vector Spaces

1. **\(\mathbb{R}^n\)**: All n-tuples of real numbers with standard addition and scalar multiplication

2. **Polynomial spaces**: \(P_n\) = {all polynomials of degree ≤ n}

3. **Function spaces**: \(C[a,b]\) = {all continuous functions on [a,b]}

4. **Matrix spaces**: \(M_{m×n}\) = {all m × n matrices}

5. **Solution spaces**: The solution set of a homogeneous system Ax = **0** forms a vector space

### 6.3 Subspaces

**Definition**: A subset W of a vector space V is a subspace if:
1. **0** ∈ W
2. If **u**, **v** ∈ W, then **u** + **v** ∈ W (closed under addition)
3. If **v** ∈ W and c ∈ F, then c**v** ∈ W (closed under scalar multiplication)

**Theorem**: W is a subspace if and only if W is nonempty and closed under linear combinations.

### 6.4 Span

**Definition**: The span of vectors \(\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}\) is the set of all possible linear combinations:

\[\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}\]

The span is always a subspace.

---

## Chapter 7: Linear Independence, Basis, and Dimension

### 7.1 Linear Independence

**Definition**: Vectors \(\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\) are linearly independent if the only solution to

\[c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}\]

is \(c_1 = c_2 = \cdots = c_k = 0\).

**Linear Dependence**: Vectors are linearly dependent if there exists a nontrivial solution (not all coefficients zero).

**Interpretation**: Vectors are linearly independent if none can be expressed as a linear combination of the others.

### 7.2 Basis

**Definition**: A basis for a vector space V is a set of vectors that:
1. Is linearly independent
2. Spans V

**Standard Basis for \(\mathbb{R}^n\)**:

\[\left\{\mathbf{e}_1 = \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}, \mathbf{e}_2 = \begin{pmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{pmatrix}, \ldots, \mathbf{e}_n = \begin{pmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{pmatrix}\right\}\]

**Key Properties**:
- Every vector in the space can be uniquely expressed as a linear combination of basis vectors
- All bases for the same vector space have the same number of elements
- If a space has a basis with k elements, then any set of more than k vectors is linearly dependent

### 7.3 Dimension

**Definition**: The dimension of a vector space V is the number of elements in any basis for V.

**Notation**: dim(V)

**Examples**:
- dim(\(\mathbb{R}^n\)) = n
- dim(\(P_n\)) = n + 1 (polynomials of degree ≤ n)
- The trivial space {\(**0**\)} has dimension 0

### 7.4 Rank and Nullity

For an m × n matrix A:

**Column Space**: The span of the columns of A

**Row Space**: The span of the rows of A

**Null Space**: The solution set of Ax = **0**

**Rank**: The dimension of the column space (= dimension of row space)

**Nullity**: The dimension of the null space

**Rank-Nullity Theorem**:
\[\text{rank}(A) + \text{nullity}(A) = n\]

where n is the number of columns.

---

## Chapter 8: Linear Transformations

### 8.1 Definition and Properties

**Definition**: A function T: V → W between vector spaces is a linear transformation if for all **u**, **v** ∈ V and all scalars c:

1. **Additivity**: T(**u** + **v**) = T(**u**) + T(**v**)
2. **Homogeneity**: T(c**v**) = cT(**v**)

**Equivalently**: T(c₁**v₁** + c₂**v₂** + ... + cₖ**vₖ**) = c₁T(**v₁**) + c₂T(**v₂**) + ... + cₖT(**vₖ**)

### 8.2 Examples of Linear Transformations

1. **Rotation** in \(\mathbb{R}^2\): Rotating by angle θ
2. **Reflection**: Reflecting across a line
3. **Projection**: Projecting onto a line or plane
4. **Scaling**: Multiplying all components by a constant
5. **Differentiation**: T(p) = p' (from polynomials to polynomials)
6. **Integration**: T(p) = ∫p(x)dx

### 8.3 Matrix Representation

**Theorem**: Every linear transformation T: \(\mathbb{R}^n\) → \(\mathbb{R}^m\) can be represented as matrix multiplication:

\[T(\mathbf{v}) = A\mathbf{v}\]

where A is an m × n matrix whose columns are T(\(\mathbf{e}_i\)) for the standard basis vectors \(\mathbf{e}_i\).

**Change of Basis**: If we use bases other than the standard basis, the matrix representation changes, but the transformation remains the same.

### 8.4 Kernel and Image

For a linear transformation T: V → W:

**Kernel (Null Space)**: 
\[\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}\]

**Image (Range)**: 
\[\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}\]

**Properties**:
- ker(T) is a subspace of V
- Im(T) is a subspace of W
- T is injective (one-to-one) if and only if ker(T) = {\(**0**\)}
- T is surjective (onto) if and only if Im(T) = W

**Rank-Nullity for Transformations**:
\[\dim(\ker(T)) + \dim(\text{Im}(T)) = \dim(V)\]

---

## Chapter 9: Eigenvalues and Eigenvectors

### 9.1 Introduction

**Problem**: For a linear transformation T represented by matrix A, find special vectors that A only scales (doesn't rotate).

**Definition**: For a square matrix A, a nonzero vector **v** is an eigenvector if

\[A\mathbf{v} = \lambda \mathbf{v}\]

for some scalar λ called the eigenvalue.

### 9.2 Finding Eigenvalues

Rearranging the eigenvalue equation:

\[A\mathbf{v} = \lambda \mathbf{v}\]
\[A\mathbf{v} - \lambda I\mathbf{v} = \mathbf{0}\]
\[(A - \lambda I)\mathbf{v} = \mathbf{0}\]

For a nontrivial solution to exist:

\[\det(A - \lambda I) = 0\]

This is the **characteristic equation**.

**Characteristic Polynomial**: det(A − λI) is a polynomial of degree n (for an n × n matrix).

The eigenvalues are the roots of this polynomial.

### 9.3 Finding Eigenvectors

Once we have an eigenvalue λ, we solve:

\[(A - \lambda I)\mathbf{v} = \mathbf{0}\]

This is a homogeneous system whose solution space is the **eigenspace** for eigenvalue λ.

### 9.4 Properties of Eigenvalues and Eigenvectors

1. **Sum of eigenvalues** = trace of A = sum of diagonal entries
2. **Product of eigenvalues** = determinant of A
3. Eigenvectors corresponding to distinct eigenvalues are linearly independent
4. An n × n matrix has at most n real eigenvalues
5. Symmetric matrices have real eigenvalues and orthogonal eigenvectors

---

## Chapter 10: Diagonalization

### 10.1 Diagonalizable Matrices

**Definition**: A square matrix A is diagonalizable if it can be written as

\[A = PDP^{-1}\]

where D is a diagonal matrix and P is invertible.

**Interpretation**: This means A represents a linear transformation that, in the basis given by the columns of P, becomes diagonal (simple scaling in each direction).

### 10.2 Diagonalization Theorem

**Theorem**: An n × n matrix A is diagonalizable if and only if it has n linearly independent eigenvectors.

**Procedure for Diagonalization**:

1. Find all eigenvalues by solving det(A − λI) = 0
2. For each eigenvalue, find a basis for the eigenspace
3. Check if we have n linearly independent eigenvectors
4. If yes, form P from these eigenvectors (as columns)
5. Form D with eigenvalues on the diagonal
6. Then A = PDP⁻¹

### 10.3 Applications of Diagonalization

**Computing Powers**:

If A = PDP⁻¹, then:
\[A^k = PD^kP^{-1}\]

Since D is diagonal, \(D^k\) is simply the diagonal entries raised to the kth power.

**Solving Recurrence Relations**:

Sequences like \(a_n = c_1 a_{n-1} + c_2 a_{n-2} + \cdots\) can be solved by writing as \(\mathbf{x}_{n+1} = A\mathbf{x}_n\) and using diagonalization.

**Matrix Exponentials**:

\[e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!} = Pe^DP^{-1}\]

### 10.4 Conditions for Diagonalizability

A matrix is NOT diagonalizable if:
- It has complex eigenvalues with multiplicity > 1 but the eigenspace has smaller dimension
- The geometric multiplicity (dimension of eigenspace) is less than the algebraic multiplicity (multiplicity as root of characteristic polynomial)

A matrix IS diagonalizable if:
- It's symmetric (for real matrices with real eigenvalues)
- All eigenvalues are distinct
- The geometric and algebraic multiplicities match for all eigenvalues

---

## Chapter 11: Inner Product Spaces

### 11.1 Inner Products

**Definition**: An inner product on a vector space V is a function ⟨·,·⟩: V × V → ℝ satisfying:

1. **Symmetry**: ⟨**u**, **v**⟩ = ⟨**v**, **u**⟩
2. **Linearity**: ⟨c**u** + d**v**, **w**⟩ = c⟨**u**, **w**⟩ + d⟨**v**, **w**⟩
3. **Positive Definiteness**: ⟨**v**, **v**⟩ > 0 for all **v** ≠ **0**

**Examples**:

1. **Standard dot product** on \(\mathbb{R}^n\): ⟨**u**, **v**⟩ = **u** · **v** = \(\sum u_iv_i\)

2. **Weighted inner product**: ⟨**u**, **v**⟩ = \(\sum w_i u_i v_i\) where \(w_i > 0\)

3. **Function space inner product**: ⟨f, g⟩ = \(\int_a^b f(x)g(x)dx\)

### 11.2 Norms and Orthogonality

**Induced Norm**:
\[\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}\]

**Orthogonal Vectors**:
Vectors **u** and **v** are orthogonal if ⟨**u**, **v**⟩ = 0

**Orthonormal Set**:
A set of vectors is orthonormal if they are mutually orthogonal and each has unit norm.

**Orthogonal Matrix**:
A square matrix Q is orthogonal if \(Q^T Q = I\), equivalently its columns form an orthonormal set.

### 11.3 Projections

**Orthogonal Projection onto a Vector**:
\[\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\langle \mathbf{v}, \mathbf{v} \rangle} \mathbf{v}\]

**Projection onto a Subspace**:
If {**v₁**, **v₂**, ..., **vₖ**} is an orthonormal basis for subspace W:
\[\text{proj}_W \mathbf{u} = \sum_{i=1}^{k} \langle \mathbf{u}, \mathbf{v}_i \rangle \mathbf{v}_i\]

### 11.4 Gram-Schmidt Orthogonalization

**Problem**: Given a linearly independent set {**v₁**, **v₂**, ..., **vₖ**}, construct an orthonormal set {**u₁**, **u₂**, ..., **uₖ**}.

**Procedure**:

1. \(\mathbf{u}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}\)

2. For i = 2 to k:
   - \(\mathbf{w}_i = \mathbf{v}_i - \sum_{j=1}^{i-1} \langle \mathbf{v}_i, \mathbf{u}_j \rangle \mathbf{u}_j\)
   - \(\mathbf{u}_i = \frac{\mathbf{w}_i}{\|\mathbf{w}_i\|}\)

This produces an orthonormal set spanning the same space.

### 11.5 QR Decomposition

Any matrix A with linearly independent columns can be written as:

\[A = QR\]

where Q has orthonormal columns and R is upper triangular.

**Application**: This decomposition is used numerically to solve least squares problems and compute eigenvalues.

### 11.6 Least Squares Solutions

For an overdetermined system Ax = b (more equations than unknowns), the least squares solution minimizes ||Ax − b||².

**Solution**: The least squares solution satisfies
\[A^T A\mathbf{x} = A^T \mathbf{b}\]

**Geometric Interpretation**: We find **x** so that Ax is the orthogonal projection of b onto the column space of A.

### 11.7 Symmetric Matrices

**Spectral Theorem**: A real symmetric matrix A can be orthogonally diagonalized:

\[A = QDQ^T\]

where Q is orthogonal and D is diagonal containing the eigenvalues.

**Consequence**: Symmetric matrices have real eigenvalues and orthogonal eigenvectors.

---

## Chapter 12: Applications and Advanced Topics

### 12.1 Data Science and Principal Component Analysis

**Principal Component Analysis (PCA)** finds directions of maximum variance in data.

**Process**:
1. Center data: Subtract mean from each observation
2. Compute covariance matrix: S = (1/n)X^T X
3. Find eigenvalues and eigenvectors of S
4. Project data onto eigenvectors (principal components)

**Application**: Data compression, feature extraction, visualization.

### 12.2 Computer Graphics and Transformations

Linear algebra underlies all computer graphics transformations:

**2D Transformations**:
- **Rotation by θ**: \(\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\)
- **Scaling by (sx, sy)**: \(\begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}\)
- **Reflection across x-axis**: \(\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}\)

**Composition**: Combining transformations through matrix multiplication.

### 12.3 Machine Learning and Neural Networks

**Neural Network Fundamentals**:
- Input layer: vector **x** ∈ \(\mathbb{R}^m\)
- Hidden layer: **h** = σ(W₁**x** + **b₁**)
- Output layer: **y** = σ(W₂**h** + **b₂**)

where W₁, W₂ are weight matrices and σ is an activation function.

**Training**: Finding optimal weights by minimizing a loss function through gradient descent (using matrix calculus).

### 12.4 Systems of Differential Equations

**System**: 
\[\frac{d\mathbf{x}}{dt} = A\mathbf{x}\]

**Solution**: If A is diagonalizable with eigenvalues λᵢ and eigenvectors **vᵢ**:
\[\mathbf{x}(t) = \sum c_i e^{\lambda_i t} \mathbf{v}_i\]

**Application**: Population dynamics, chemical reactions, mechanical systems.

### 12.5 Fourier Series and Harmonic Analysis

**Orthonormal Basis**: Functions {1, cos(x), sin(x), cos(2x), sin(2x), ...} form an orthonormal basis for function spaces (with appropriate inner product).

**Fourier Series**: Any periodic function can be expressed as:
\[f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} [a_n \cos(nx) + b_n \sin(nx)]\]

**Linear Algebra Connection**: This is expressing f as a linear combination of basis functions.

### 12.6 Cryptography

**Hill Cipher**: An encryption method using matrices.

- **Encryption**: **c** = A**p** (mod m)
- **Decryption**: **p** = A⁻¹**c** (mod m)

**Security**: Based on the difficulty of computing matrix inverses without knowing the key.

### 12.7 Error-Correcting Codes

**Linear Codes**: Encode messages using generator and parity-check matrices.

- **Generator Matrix**: G produces codewords
- **Parity-Check Matrix**: H used to detect and correct errors

**Hamming Codes**: Classic error-correcting code using linear algebra.

---

## Conclusion

Linear algebra is a fundamental mathematical framework with remarkable breadth and depth. From solving simple systems of equations to powering modern artificial intelligence, linear algebra provides the essential language for organized thinking about multidimensional problems.

The key concepts—vectors, matrices, eigenvalues, and vector spaces—form an interconnected web of powerful ideas. Mastering these concepts opens doors to advanced mathematics, computer science, physics, engineering, and data science.

**Further Study**:
- Advanced topics: Tensor analysis, functional analysis, abstract algebra
- Numerical methods: Algorithms for computing eigenvalues and solving large systems
- Applications: Deeper study in machine learning, computational physics, signal processing

The journey through linear algebra continues to reveal beautiful connections between abstract theory and concrete applications.

---

## Appendix A: Quick Reference

### Common Definitions and Theorems

| Concept | Definition |
|---------|-----------|
| Vector | Ordered array of numbers; elements of \(\mathbb{R}^n\) |
| Matrix | Rectangular array of numbers |
| Linear Transformation | Function T: V → W with T(**u** + **v**) = T(**u**) + T(**v**) and T(c**v**) = cT(**v**) |
| Eigenvalue | Scalar λ such that A**v** = λ**v** for nonzero **v** |
| Eigenvector | Nonzero vector **v** satisfying A**v** = λ**v** |
| Basis | Linearly independent spanning set |
| Dimension | Number of elements in any basis |
| Determinant | Scalar encoding invertibility and volume scaling |
| Rank | Dimension of column space |
| Inner Product | Generalization of dot product satisfying symmetry, linearity, positive definiteness |

### Matrix Properties Quick Guide

- **Invertible if**: det(A) ≠ 0, full rank, all eigenvalues nonzero
- **Diagonalizable if**: Has n linearly independent eigenvectors (n × n matrix)
- **Orthogonal if**: Columns are orthonormal, Q^T Q = I
- **Symmetric if**: A^T = A (real eigenvalues, orthogonal eigenvectors)

---

## Appendix B: Practice Problems and Solutions

[Complete problem sets with solutions would follow in a full textbook]

---

## Bibliography and Further Reading

1. Axler, S. (2015). *Linear Algebra Done Right*. Springer.
2. Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
3. Selinger, P. (2018). *Linear Algebra*. Open text.
4. Cherney, D., Denton, T., Thomas, R., & Waldron, A. (2013). *Linear Algebra*. UC Davis.
5. Horn, R. A., & Johnson, C. R. (2012). *Matrix Analysis*. Cambridge University Press.

---

**End of Comprehensive Linear Algebra Book**