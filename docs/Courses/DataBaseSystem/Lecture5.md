# Entity-Relationship Model
## Preliminary
### Steps of Database Design
1. conceptual database design 
> (choose data model, i.e. E-R model, Net Model)
3. Logical database design
> convert the conceptual design into a DB schema
4. Schema refinement
> Normalization of relations: Check relational schema for redundancies and related anomalies(异常现象). 
5. Physical database design
> indexing, query, clustering, and database tuning
6. Create and initialize the database & Security design 
> Load initial data, testing
> Identify different user groups and their roles

![alt text](image-12.png)

### Modeling
![alt text](image-13.png)

## Entity Sets
关于 Entity 要注意的是 它的属性 Attributes 的不同种类
![alt text](image-14.png)

## Relationship Sets
类比数据库中的关系表，或者说数据库中的关系表就是由此而来的
## Keys for Enitity Sets
![alt text](image-15.png)
![alt text](image-16.png)

## E-R Diagram
![alt text](image-17.png)
### 用图表示
![alt text](image-18.png)
或
![alt text](image-19.png)
### 要点、特性
- Entity sets of a relationship need not be distinct, e.g., Recursive relationship set (自环联系集)
![alt text](image-28.png)
- the Cardinality Constraints 
  - We express cardinality constraints by drawing either a directed line (->), signifying “one”, or an undirected line (—), signifying “many”,  between the relationship set and the entity set. 
- Participation of an Entity Set in a Relationship Set (单横线与双横线)
  - ![alt text](image-20.png)
- Alternative Notation for relationship Constraints 
  - 语文题，略
- E-R Diagram with a Ternary Relationship 
  - ![alt text](image-21.png)
  - Binary v.s. Ternary
  ![alt text](image-22.png)
  - Converting Non-Binary to Binary Form
  > 人为插入 entity set 或观察选取其中一个
    ![alt text](image-23.png)
    ![alt text](image-24.png)

## Weak Entity Sets
- An entity set that does not have a primary key
- The existence of a weak entity set **depends on** the existence of a identifying entity set or owner entity set (标识实体集或属主实体集)
  - It **must** relate to the identifying entity set **via a total, one-to-many relationship set** from the identifying to the weak entity set
  - Identifying relationship depicted using a double diamond 在E-R图上用双重钻石框表示
  - The discriminator or partial key (分辨符或部分码) of a weak entity set is the set of attributes that distinguishes among all those entities in a weak entity set that depend on one particular strong entity (e.g., 例2中的payment-number).
  - The primary key of a weak entity set is formed by the primary key of the strong entity set on which the weak entity set is existence dependent, plus the weak entity set’s discriminator. 
  
  
!!! example example 1
    ![alt text](image-25.png)


!!! example example 2
    ![alt text](image-26.png)
  
## 小结
!!! example 
    ![alt text](image-27.png)

## Extended E-R Feature
### Stratum of the entity set 
#### Specialization (特殊化、具体化) 
![alt text](image-29.png)
!!! example 
    ![alt text](image-30.png)
#### Generalization (泛化、普遍化) 
![alt text](image-31.png)
!!! example

#### Design Constraints on a Specialization / Generalization (Cont.) 
待补充

#### Aggregation 
待补充

## Summary of Symbols Used in E-R Notation
![alt text](image-32.png)
![alt text](image-33.png)
![alt text](image-34.png)
![alt text](image-35.png)

## Design of an E-R Database Schema

## Reduction of an E-R Schema to Tables

